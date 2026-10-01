"""Spectral-conv variants that reshape the matmuls: block-fused mix, Karatsuba mix, stacked cos/sin."""
import jax.numpy as jnp

from ...fft.core import ACC, F
from ...fno.spectral import _dft_rows

_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)


def _bands(n_rows, n_cols, n_modes):
    half = n_modes // 2
    row_band = jnp.concatenate([jnp.arange(half), jnp.arange(n_rows - half, n_rows)])
    return half, row_band


def _inverse(mixed_re, mixed_im, half, row_band, n_rows, n_cols):
    """Shared tail: columns back with the Hermitian fold, then rows back, real part only."""
    col_modes = jnp.arange(half)
    fold = jnp.where(col_modes == 0, 1.0, 2.0).astype(F)
    inv_col_cos, inv_col_sin = _dft_rows(col_modes, n_cols, +1)
    inv_col_cos = (inv_col_cos * fold[:, None]).T
    inv_col_sin = (inv_col_sin * fold[:, None]).T
    back = lambda m, z: _E("wd,noad->noaw", m, z)
    cols_re = back(inv_col_cos, mixed_re) - back(inv_col_sin, mixed_im)
    cols_im = back(inv_col_cos, mixed_im) + back(inv_col_sin, mixed_re)
    inv_row_cos, inv_row_sin = _dft_rows(row_band, n_rows, +1)
    rows = lambda m, z: _E("ha,noaw->nohw", m.T, z)
    field = rows(inv_row_cos, cols_re) - rows(inv_row_sin, cols_im)
    return (field / (n_rows * n_cols)).astype(F)


def _forward(x, half, row_band, n_rows, n_cols, dtype, stacked):
    """Forward transform. stacked=True does cos and sin in one matmul by concatenating the two
    matrices along their output axis, so the MXU sees one call with twice the output rows."""
    if stacked:
        rc, rs = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
        both = _E("ah,nchw->ncaw", jnp.concatenate([rc, rs], axis=0), x)
        a = row_band.shape[0]
        rows_re, rows_im = both[:, :, :a], both[:, :, a:]

        cc, cs = (m.astype(dtype) for m in _dft_rows(jnp.arange(half), n_cols, -1))
        colm = jnp.concatenate([cc, cs], axis=0)
        both_c = _E("dw,ncaw->ncad", colm, rows_re)
        both_d = _E("dw,ncaw->ncad", colm, rows_im)
        cr, ci = both_c[..., :half], both_c[..., half:]
        dr, di = both_d[..., :half], both_d[..., half:]
        return cr - di, ci + dr

    rc, rs = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
    rows_re, rows_im = _E("ah,nchw->ncaw", rc, x), _E("ah,nchw->ncaw", rs, x)
    cc, cs = (m.astype(dtype) for m in _dft_rows(jnp.arange(half), n_cols, -1))
    ov = lambda m, z: _E("dw,ncaw->ncad", m, z)
    return ov(cc, rows_re) - ov(cs, rows_im), ov(cc, rows_im) + ov(cs, rows_re)


def _mix_naive(w_re, w_im, zr, zi):
    mix = lambda w, z: _E("adio,niad->noad", w, z)
    return mix(w_re, zr) - mix(w_im, zi), mix(w_re, zi) + mix(w_im, zr)


def _mix_block(w_re, w_im, zr, zi):
    """One matmul instead of four: [[Wre,-Wim],[Wim,Wre]] against [Zre;Zim], stacking both the
    contracted channel axis and the output axis, so the MXU sees 2*channels on each side."""
    left = jnp.concatenate([w_re, -w_im], axis=2)     # (a, d, 2C, C) -> real part
    right = jnp.concatenate([w_im, w_re], axis=2)     # (a, d, 2C, C) -> imag part
    w = jnp.concatenate([left, right], axis=3)        # (a, d, 2C, 2C)
    z = jnp.concatenate([zr, zi], axis=1)             # (n, 2C, a, d)
    out = _E("adio,niad->noad", w, z)
    c = w_re.shape[3]
    return out[:, :c], out[:, c:]


def _mix_karatsuba(w_re, w_im, zr, zi):
    """Three matmuls instead of four (Gauss/Karatsuba)."""
    mix = lambda w, z: _E("adio,niad->noad", w, z)
    a = mix(w_re, zr)
    b = mix(w_im, zi)
    c = mix(w_re + w_im, zr + zi)
    return a - b, c - a - b


MIXES = {"naive": _mix_naive, "block": _mix_block, "karatsuba": _mix_karatsuba}


def spectral_conv_variant(weight, x, n_modes, dtype=F, mix="naive", stacked=False):
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    half, row_band = _bands(n_rows, n_cols, n_modes)
    x = x.astype(dtype)
    zr, zi = _forward(x, half, row_band, n_rows, n_cols, dtype, stacked)
    w_re, w_im = weight["w_re"].astype(dtype), weight["w_im"].astype(dtype)
    mr, mi = MIXES[mix](w_re, w_im, zr.astype(dtype), zi.astype(dtype))
    return _inverse(mr, mi, half, row_band, n_rows, n_cols)
