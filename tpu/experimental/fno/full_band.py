"""Spectral conv keeping the full mode band instead of half plus a conjugate fold."""
import jax.numpy as jnp

from ...fft.core import ACC, F
from ...fno.spectral import _dft_rows

_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)


def spectral_conv_full(weight, x, n_modes, dtype=F):
    """Same function as spectral_conv, but the column axis carries all n_modes wavenumbers
    (both signs) rather than the non-negative half. The mirrored half of the weight is the
    conjugate of the kept half, so the model is unchanged -- only the work differs."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    half = n_modes // 2
    n_ch = weight["w_re"].shape[1]
    x = x.astype(dtype)

    row_band = jnp.concatenate([jnp.arange(half), jnp.arange(n_rows - half, n_rows)])
    col_band = jnp.concatenate([jnp.arange(half), jnp.arange(n_cols - half, n_cols)])
    band = row_band.shape[0]

    rc, rs = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
    rows_both = _E("ah,nchw->ncaw", jnp.concatenate([rc, rs], axis=0), x)
    rows_re, rows_im = rows_both[:, :, :band], rows_both[:, :, band:]

    cc, cs = (c.astype(dtype) for c in _dft_rows(col_band, n_cols, -1))
    col_both = jnp.concatenate([cc, cs], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col_both, z)
    fr, fi = over(rows_re), over(rows_im)
    wide = col_band.shape[0]
    modes_re = fr[..., :wide] - fi[..., wide:]
    modes_im = fr[..., wide:] + fi[..., :wide]

    # weight over the full band: the mirrored (-k1,-k2) entry is the conjugate of (k1,k2)
    w_re, w_im = weight["w_re"].astype(dtype), weight["w_im"].astype(dtype)   # (i,o,band,half)
    mir_re = jnp.flip(jnp.roll(w_re, -1, axis=2), axis=(2, 3))
    mir_im = -jnp.flip(jnp.roll(w_im, -1, axis=2), axis=(2, 3))
    full_re = jnp.concatenate([w_re, mir_re], axis=3)                         # (i,o,band,2*half)
    full_im = jnp.concatenate([w_im, mir_im], axis=3)

    w_block = jnp.concatenate([jnp.concatenate([full_re, -full_im], axis=0),
                               jnp.concatenate([full_im, full_re], axis=0)], axis=1)
    z_block = jnp.concatenate([modes_re, modes_im], axis=1)
    mixed = _E("ioad,niad->noad", w_block, z_block)
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    icc, ics = _dft_rows(col_band, n_cols, +1)                                # no fold needed
    icc, ics = icc.T, ics.T
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    cols_back = _E("wd,noad->noaw", inv_col, jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]

    irc, irs = _dft_rows(row_band, n_rows, +1)
    inv_row = jnp.concatenate([irc, -irs], axis=0)
    field = _E("ha,noaw->nohw", inv_row.T, jnp.concatenate([cb_re, cb_im], axis=2))
    return (field / (n_rows * n_cols)).astype(F)
