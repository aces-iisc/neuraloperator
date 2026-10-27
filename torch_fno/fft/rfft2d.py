import torch

from .core import ACC, F, complex_matmul, dft_matrix_reim, rfft_staged, rows_matrix

# Contract the middle axis of an (a, w, t) block against a (d, w) matrix -- no transpose.
_mix_axis1 = lambda mat, x: torch.einsum("dw,awt->adt", mat, x).to(ACC)


def rfft2_full(field, dtype=F):
    """Full 2-D rfft of a real (n_rows, n_cols, batch) field. Returns (re, im)."""
    n_rows, n_cols, batch = field.shape
    rows_re, rows_im = rfft_staged(field.reshape(n_rows, n_cols * batch), n_rows,
                                   half=True, dtype=dtype)
    kept = rows_re.shape[0]
    rows_re = rows_re.reshape(kept, n_cols, batch)
    rows_im = rows_im.reshape(kept, n_cols, batch)
    cos, sin, cos_plus_sin = dft_matrix_reim(n_cols, dtype, field.device)
    return complex_matmul(_mix_axis1, cos, sin, cos_plus_sin, rows_re, rows_im)


def rfft2_low_modes(field, ratio=0.5, dtype=F):
    """Compute only the lowest modes on each axis as a direct partial DFT.

    No full FFT, no twiddle, no transpose: the alternative builds the whole spectrum and throws
    most of it away. Real (n_rows, n_cols, batch) -> (re, im), each (row_modes, col_modes, batch).
    """
    n_rows, n_cols, batch = field.shape
    row_modes = max(1, int(n_rows * ratio))
    col_modes = max(1, int(n_cols * ratio))
    device = field.device

    row_cos, row_sin = rows_matrix(range(row_modes), n_rows, -1, dtype, device)
    flat = field.reshape(n_rows, n_cols * batch).to(dtype)
    rows_re = torch.matmul(row_cos, flat).to(ACC).reshape(row_modes, n_cols, batch)
    rows_im = torch.matmul(row_sin, flat).to(ACC).reshape(row_modes, n_cols, batch)

    col_cos, col_sin = rows_matrix(range(col_modes), n_cols, -1, dtype, device)
    return complex_matmul(_mix_axis1, col_cos, col_sin, (col_cos + col_sin).to(dtype),
                          rows_re, rows_im)
