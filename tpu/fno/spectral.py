import jax.numpy as jnp

from ..fft.core import ACC, F

# einsum letters: n=sample c/i=in-channel o=out-channel h=row w=col
#                 a=row mode (the 2m band)  d=col mode (the low m)
_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)


def _dft_rows(mode_idx, length, sign):
    """(cos, sin) rows of a length-`length` DFT restricted to the mode indices `mode_idx`.
    sign=-1 builds the forward transform, +1 the inverse.  Each is (len(mode_idx), length)."""
    k = jnp.asarray(mode_idx)[:, None]
    n = jnp.arange(length)[None, :]
    angle = (sign * 2.0 * jnp.pi / length) * (k * n).astype(F)
    return jnp.cos(angle), jnp.sin(angle)


def spectral_conv(weight, x, n_modes, dtype=F):
    """One FNO spectral convolution, done entirely in real/imag — no complex64 anywhere.

    Four steps, each a matmul over one axis:
      1. transform the rows, keeping only the low +/- band  (real input, so 2 matmuls)
      2. transform the cols, keeping only the low modes     (complex now, so 4)
      3. mix channels independently at every kept mode      (the learned weight)
      4. transform both axes back and take the real part

    Steps 1-2 compute ONLY the modes that survive truncation, so no full FFT ever happens, and
    step 4 reconstructs the full field straight from them — the un-kept modes are implicitly zero,
    with no padding tensor.  `n_modes` is the TOTAL band width, so each corner holds n_modes//2.
    `weight` is {w_re, w_im} of shape (in_ch, out_ch, n_modes, n_modes//2).
    `dtype` is the operand precision the transform is carried in: F is the f32 default, BF halves
    the activation bytes on a memory-bound kernel.  Accumulation stays f32 either way.
    x (B, in_ch, H, W) real -> (B, out_ch, H, W) real."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    half = n_modes // 2          # n_modes counts BOTH corners; each corner holds half of them
    n_ch = weight["w_re"].shape[1]
    x = x.astype(dtype)

    # 1. rows -> both low-frequency corners, n_modes wavenumbers in total.
    #    cos and sin hit the SAME x, so they go as one matmul with the two matrices stacked
    #    along the output axis; the MXU sees twice the rows per call.
    row_band = jnp.concatenate([jnp.arange(half), jnp.arange(n_rows - half, n_rows)])
    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
    band = row_band.shape[0]
    rows_both = _E("ah,nchw->ncaw", jnp.concatenate([row_cos, row_sin], axis=0), x)
    rows_re, rows_im = rows_both[:, :, :band], rows_both[:, :, band:]

    # 2. cols -> the low n_modes only (this axis is rfft'd, so negative cols are redundant),
    #    again with cos and sin stacked into one matmul per operand
    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(jnp.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over_cols = lambda z: _E("dw,ncaw->ncad", col_both, z)
    from_re, from_im = over_cols(rows_re), over_cols(rows_im)
    modes_re = from_re[..., :half] - from_im[..., half:]
    modes_im = from_re[..., half:] + from_im[..., :half]

    # 3. learned complex channel mix, per kept mode, as ONE real block matmul:
    #    [[w_re, -w_im], [w_im, w_re]] against [modes_re; modes_im].  Same arithmetic as the
    #    four separate products, but one call and twice the contracted channels on the MXU.
    w_re, w_im = weight["w_re"].astype(dtype), weight["w_im"].astype(dtype)
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1)
    z_block = jnp.concatenate([modes_re.astype(dtype), modes_im.astype(dtype)], axis=1)
    mixed = _E("ioad,niad->noad", w_block, z_block)
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    # 4a. cols back, folding in the Hermitian partners the rfft'd axis left out
    #     (DC counts once, every other mode twice — that is its -k conjugate)
    col_modes = jnp.arange(half)
    hermitian_fold = jnp.where(col_modes == 0, 1.0, 2.0).astype(F)
    inv_col_cos, inv_col_sin = _dft_rows(col_modes, n_cols, +1)
    inv_col_cos = (inv_col_cos * hermitian_fold[:, None]).T          # (n_cols, n_modes)
    inv_col_sin = (inv_col_sin * hermitian_fold[:, None]).T
    #     same block form as step 3: [[cos, -sin], [sin, cos]] against [re ; im], one matmul
    inv_col_block = jnp.concatenate(
        [jnp.concatenate([inv_col_cos, -inv_col_sin], axis=1),
         jnp.concatenate([inv_col_sin, inv_col_cos], axis=1)], axis=0)      # (2*n_cols, 2*half)
    mixed_block = jnp.concatenate([mixed_re, mixed_im], axis=-1)            # (..., 2*half)
    cols_back = _E("wd,noad->noaw", inv_col_block, mixed_block)
    cols_back_re, cols_back_im = cols_back[..., :n_cols], cols_back[..., n_cols:]

    # 4b. rows back over the same band; the field is real, so only the real part survives, so
    #     the cosine and sine halves fold into a single matmul rather than two
    inv_row_cos, inv_row_sin = _dft_rows(row_band, n_rows, +1)
    inv_row_block = jnp.concatenate([inv_row_cos, -inv_row_sin], axis=0)    # (2*band, n_rows)
    cols_block = jnp.concatenate([cols_back_re, cols_back_im], axis=2)      # (..., 2*band, ...)
    field = _E("ha,noaw->nohw", inv_row_block.T, cols_block)

    return (field / (n_rows * n_cols)).astype(F)
