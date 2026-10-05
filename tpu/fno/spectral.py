import jax.numpy as jnp
import numpy as np

from ..fft.core import ACC, F
from ..fft.transform import FFT
from .layers import spectral_block

# einsum letters: n=sample c/i=in-channel o=out-channel h=row w=col
#                 a=row mode (the 2m band)  d=col mode (the low m)
_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)

DEFAULT_FFT = FFT()          # TPU v5e; pass another to retarget without editing the layer


def _dft_rows(mode_idx, length, sign):
    """(cos, sin) rows of a length-`length` DFT restricted to the mode indices `mode_idx`.
    sign=-1 builds the forward transform, +1 the inverse.  Each is (len(mode_idx), length).

    Kept for the benches that build matrices directly; the layer goes through FFT."""
    k = jnp.asarray(mode_idx)[:, None]
    n = jnp.arange(length)[None, :]
    angle = (sign * 2.0 * jnp.pi / length) * (k * n).astype(F)
    return jnp.cos(angle), jnp.sin(angle)


def spectral_conv(weight, x, n_modes, dtype=F, fft=None):
    """One FNO spectral convolution, done entirely in real/imag — no complex64 anywhere.

    Four steps, each a matmul over one axis:
      1. transform the rows, keeping only the low +/- band  (real input, so 2 matmuls stacked)
      2. transform the cols, keeping only the low modes     (complex now)
      3. mix channels independently at every kept mode      (the learned weight)
      4. transform both axes back and take the real part

    Steps 1-2 compute ONLY the modes that survive truncation, so no full FFT ever happens, and
    step 4 reconstructs the full field straight from them — the un-kept modes are implicitly
    zero, with no padding tensor.  `n_modes` is the TOTAL band width, so each corner holds
    n_modes//2.  `weight` is {w_re, w_im} of shape (n_modes, n_modes//2, in_ch, out_ch), or a
    prebuilt {w_block} from layers.prebuild_blocks.
    `dtype` is the operand precision the transform is carried in: F is the f32 default, BF
    halves the activation bytes on a memory-bound kernel.  Accumulation stays f32 either way.
    `fft` selects the transform, and with it the accelerator it is shaped for.
    x (B, in_ch, H, W) real -> (B, out_ch, H, W) real."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    half = n_modes // 2          # n_modes counts BOTH corners; each corner holds half of them
    n_ch = (weight["w_block"].shape[3] // 2 if "w_block" in weight
            else weight["w_re"].shape[3])      # (a, d, in, out): out is the minor axis
    transform = fft if fft is not None else (DEFAULT_FFT if dtype is DEFAULT_FFT.dtype
                                             else FFT(DEFAULT_FFT.accel, dtype))
    row_band = transform.band(n_modes, n_rows)
    col_band = transform.half_band(n_modes)

    # 1-2. rows to both low-frequency corners, then cols to the low modes.  Each pass contracts
    #      its own axis in full and truncates only its OUTPUT, so the two truncations compound.
    rows_re, rows_im = transform.rows_forward(x, row_band)
    modes_re, modes_im = transform.cols_forward(rows_re, rows_im, col_band, n_cols)

    # 3. learned complex channel mix, per kept mode, as ONE real block matmul:
    #    [[w_re, -w_im], [w_im, w_re]] against [modes_re; modes_im].  Same arithmetic as the
    #    four separate products, but one call and twice the contracted channels on the MXU.
    w_block = weight.get("w_block")
    if w_block is None:                       # not prebuilt: fall back to building it here
        w_block = spectral_block(weight["w_re"], weight["w_im"], dtype)
    elif w_block.dtype != dtype:
        w_block = w_block.astype(dtype)
    z_block = jnp.concatenate([modes_re.astype(dtype), modes_im.astype(dtype)], axis=1)
    mixed = _E("adio,niad->noad", w_block, z_block)
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    # 4. both axes back; the column pass folds in the Hermitian partners the one-sided axis
    #    left out, and the row pass keeps only the real part.
    cols_re, cols_im = transform.cols_inverse(mixed_re, mixed_im, col_band, n_cols)
    field = transform.rows_inverse(cols_re, cols_im, row_band, n_rows)
    return (field / (n_rows * n_cols)).astype(F)


def separable_conv(weight, x, n_modes, dtype=F, fft=None):
    """F-FNO's spectral convolution: one weight per axis instead of one per mode pair.

    The dense filter carries a weight for every (channel pair, mode PAIR).  This carries one per
    channel pair and AXIS, and adds the two results, so the m*m/2 mode cells collapse to m row
    modes plus m/2 column modes.  Weights are {kr, ki, lr, li}, each (modes, in_ch, out_ch),
    blocked and stacked the same way the dense filter is.

    This changes the model, not just how it is evaluated: it is a different parameterisation
    with its own accuracy, so it does not reproduce the dense output.
    """
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    n_ch = weight["k_block"].shape[2] // 2
    transform = fft if fft is not None else (DEFAULT_FFT if dtype is DEFAULT_FFT.dtype
                                             else FFT(DEFAULT_FFT.accel, dtype))
    row_band = transform.band(n_modes, n_rows)
    col_band = transform.half_band(n_modes)

    rows_re, rows_im = transform.rows_forward(x, row_band)
    modes_re, modes_im = transform.cols_forward(rows_re, rows_im, col_band, n_cols)

    # one contraction per axis, summed: W[a, d] = W_row[a] + W_col[d] is an outer sum, so the
    # two weights never have to be combined into a mode-pair grid.
    z = jnp.concatenate([modes_re.astype(dtype), modes_im.astype(dtype)], axis=1)
    mixed = (_E("aio,niad->noad", weight["k_block"].astype(dtype), z)
             + _E("dio,niad->noad", weight["l_block"].astype(dtype), z))
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    cols_re, cols_im = transform.cols_inverse(mixed_re, mixed_im, col_band, n_cols)
    field = transform.rows_inverse(cols_re, cols_im, row_band, n_rows)
    return (field / (n_rows * n_cols)).astype(F)


def cp_conv(weight, x, n_modes, dtype=F, fft=None):
    """TFNO's spectral convolution: a rank-R CP filter in place of the dense one.

    Steps 1, 2, 4 and 5 are the transform and are identical to spectral_conv; only the channel
    mix differs.  The contraction runs over the rank axis, and R is small by construction, so it
    is issued the way Section 5.2 issues the dense product rather than as four separate rank-R
    applications: the input halves are stacked on the channel axis so one contraction emits the
    whole 4R operand, and the four output products are one (2*out, 4R) matmul.

    Like separable_conv this is a different parameterisation, not another way of evaluating the
    dense filter, so it does not reproduce the dense output.
    """
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    n_ch = weight["re_B"].shape[0]
    rank = weight["re_A"].shape[1]
    transform = fft if fft is not None else (DEFAULT_FFT if dtype is DEFAULT_FFT.dtype
                                             else FFT(DEFAULT_FFT.accel, dtype))
    row_band = transform.band(n_modes, n_rows)
    col_band = transform.half_band(n_modes)

    rows_re, rows_im = transform.rows_forward(x, row_band)
    modes_re, modes_im = transform.cols_forward(rows_re, rows_im, col_band, n_cols)

    # one contraction over the stacked input channels emits [a, b, c, d] in rank order
    a_stack = jnp.concatenate([weight["re_A"].T, weight["im_A"].T], axis=0)   # (2r, i)
    zero = jnp.zeros_like(a_stack)
    a_big = jnp.concatenate([jnp.concatenate([a_stack, zero], axis=1),
                             jnp.concatenate([zero, a_stack], axis=1)], axis=0)   # (4r, 2i)
    cd_re = weight["re_C"].T[:, :, None] * weight["re_D"].T[:, None, :]           # (r, a, d)
    cd_im = weight["im_C"].T[:, :, None] * weight["im_D"].T[:, None, :]
    cd = jnp.concatenate([cd_re, cd_im, cd_re, cd_im], axis=0)                    # (4r, a, d)
    zeros_b = jnp.zeros_like(weight["re_B"])
    b_block = jnp.concatenate([
        jnp.concatenate([weight["re_B"], zeros_b, zeros_b, -weight["im_B"]], axis=1),
        jnp.concatenate([zeros_b, weight["im_B"], weight["re_B"], zeros_b], axis=1)], axis=0)

    z = jnp.concatenate([modes_re.astype(dtype), modes_im.astype(dtype)], axis=1)
    v = _E("ri,niad->nrad", a_big.astype(dtype), z)
    v = v * cd.astype(ACC)[None]
    mixed = _E("or,nrad->noad", b_block.astype(dtype), v)
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    cols_re, cols_im = transform.cols_inverse(mixed_re, mixed_im, col_band, n_cols)
    field = transform.rows_inverse(cols_re, cols_im, row_band, n_rows)
    return (field / (n_rows * n_cols)).astype(F)
