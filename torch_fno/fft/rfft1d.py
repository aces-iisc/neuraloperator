import torch

from .core import ACC, F, rfft_staged, stacked_rows


def rfft_full(signal):
    """Full n-bin FFT of a real (n, batch) input, carried entirely as real/imag."""
    return rfft_staged(signal.to(F), signal.shape[0], half=False)


def rfft_half(signal, dtype=F):
    """The n//2 non-redundant bins of a real (n, batch) input."""
    return rfft_staged(signal, signal.shape[0], half=True, dtype=dtype)


def rfft_rows(signal, rows, dtype=F):
    """The DFT bins listed in `rows`, evaluated directly: the matrix is (len(rows), n).

    cos and sin hit the same operand, so they go as one matmul with the two stacked along the
    output axis. Exact, not a truncation -- output frequencies are independent.
    """
    matrix = stacked_rows(rows, signal.shape[0], -1, dtype, signal.device)
    both = torch.matmul(matrix, signal.to(dtype)).to(ACC)
    keep = len(rows)
    return both[:keep], both[keep:]


def rfft_low_modes(signal, n_modes, dtype=F):
    """The lowest `n_modes` bins as one direct (2*n_modes, n) matmul: one pass, no twiddle."""
    return rfft_rows(signal, range(n_modes), dtype)


def rfft_direct(signal, half=True, dtype=F):
    """One direct DFT. O(n^2) but every tile is full, which wins at small n."""
    n = signal.shape[0]
    return rfft_rows(signal, range(n // 2 if half else n), dtype)
