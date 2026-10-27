import jax.numpy as jnp

from .core import ACC, F, complex_matmul, dft_matrix_reim, fft_staged_complex, rfft_staged

# Contract the middle axis of an (a, w, t) block against a (d, w) matrix -- no transpose.
_mix_axis1 = lambda mat, x: jnp.einsum('dw,awt->adt', mat, x, preferred_element_type=ACC)


def _dft_axis1(x_re, x_im, size, dtype=F):
    """Complex DFT of length `size` along axis 1 of (rows, size, batch), as a direct size x size
    matmul contracting the middle axis."""
    cos, sin, cos_plus_sin = dft_matrix_reim(size, dtype)
    return complex_matmul(_mix_axis1, cos, sin, cos_plus_sin, x_re, x_im)


def rfft2_full(field, dtype=F):
    """Full 2-D rfft of a real (n_rows, n_cols, batch) field.  Rows go through the shared staged
    real pass; cols through a direct complex DFT on the middle axis, so nothing is transposed.
    Returns (re, im), each (n_rows//2, n_cols, batch)."""
    n_rows, n_cols, batch = field.shape
    rows_re, rows_im = rfft_staged(field.reshape(n_rows, n_cols * batch), n_rows,
                                   half=True, dtype=dtype)
    kept = rows_re.shape[0]                                   # n_rows // 2
    rows_re = rows_re.reshape(kept, n_cols, batch)
    rows_im = rows_im.reshape(kept, n_cols, batch)
    return _dft_axis1(rows_re, rows_im, n_cols, dtype=dtype)


def rfft2_full_staged_cols(field, dtype=F):
    """Same as rfft2_full, but the column transform is a STAGED radix FFT (O(n log n)) instead of
    the direct O(n^2) DFT.  Staging needs the axis up front, so this pays two transposes — it wins
    only when n_cols is large enough that the quadratic column DFT dominates that cost."""
    n_rows, n_cols, batch = field.shape
    rows_re, rows_im = rfft_staged(field.reshape(n_rows, n_cols * batch), n_rows,
                                   half=True, dtype=dtype)
    kept = rows_re.shape[0]
    to_front = lambda a: a.reshape(kept, n_cols, batch).transpose(1, 0, 2).reshape(n_cols, kept * batch)
    cols_re, cols_im = fft_staged_complex(to_front(rows_re).astype(dtype),
                                          to_front(rows_im).astype(dtype), n_cols)
    back = lambda a: a.reshape(n_cols, kept, batch).transpose(1, 0, 2)
    return back(cols_re), back(cols_im)


def rfft2_low_modes(field, ratio=0.5, dtype=F):
    """The FNO transform, and the measured champion: compute ONLY the lowest modes on each axis as
    a direct partial DFT.  No full FFT, no twiddle, no transpose, no complex64 — 4.2x faster than
    doing a full jnp.rfft2 and throwing most of it away (results/REPORT.md §2).

    Real (n_rows, n_cols, batch) -> (re, im), each (row_modes, col_modes, batch)."""
    n_rows, n_cols, batch = field.shape
    row_modes = max(1, int(n_rows * ratio))
    col_modes = max(1, int(n_cols * ratio))

    # Rows: partial real DFT, real input so 2 matmuls.
    row_angle = -2 * jnp.pi * jnp.outer(jnp.arange(row_modes), jnp.arange(n_rows)) / n_rows
    row_cos, row_sin = jnp.cos(row_angle).astype(dtype), jnp.sin(row_angle).astype(dtype)
    flat = field.reshape(n_rows, n_cols * batch).astype(dtype)
    rows_re = jnp.matmul(row_cos, flat, preferred_element_type=ACC).reshape(row_modes, n_cols, batch)
    rows_im = jnp.matmul(row_sin, flat, preferred_element_type=ACC).reshape(row_modes, n_cols, batch)

    # Cols: partial complex DFT, Karatsuba, contracting the middle axis in place.
    col_angle = -2 * jnp.pi * jnp.outer(jnp.arange(col_modes), jnp.arange(n_cols)) / n_cols
    col_cos, col_sin = jnp.cos(col_angle).astype(dtype), jnp.sin(col_angle).astype(dtype)
    return complex_matmul(_mix_axis1, col_cos, col_sin,
                          (col_cos + col_sin).astype(dtype), rows_re, rows_im)
