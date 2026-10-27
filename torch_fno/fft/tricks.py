"""The two real-input packing schemes, as a torch port of the TPU engines.

Both exploit the same fact as the half-spectrum transform -- a real signal's spectrum is
Hermitian, so half the output is redundant -- but they spend the redundancy differently:

  coupled   pair two real signals into one complex signal a + ib, transform once, separate the
            two spectra afterwards.  Half as many transforms, each on complex data.
  reduced   split one signal into its even and odd samples, transform both at half length, and
            recombine with a twiddle.  Same number of transforms, each half as long.

Neither is the default anywhere. On a TPU they lose to the plain real/imag partial transform by
10-20% at full length, and across a band the three land within a few percent of each other, which
is inside that measurement. They are ported here because the GPU is a different machine and the
question is empirical: the packing changes the matmul's shape, and what a shape is worth is the
one thing that does not carry across hardware.

Nothing in the model imports this module; it exists for the engine bench.
"""
import functools
import math

import torch

from .core import ACC, F, fft_staged_complex, stacked_rows


@functools.lru_cache(maxsize=256)
def _twiddle(n_modes, length, dtype, device):
    """W_N^k for the first `n_modes` bins, as a column so it broadcasts over the signals."""
    k = torch.arange(n_modes, device=device, dtype=F)
    angle = (-2.0 * math.pi / length) * k
    return torch.cos(angle).to(dtype)[:, None], torch.sin(angle).to(dtype)[:, None]


def rfft_coupled_low_modes(signal, n_modes, dtype=F):
    """Pair the K real signals into K/2 complex ones, then separate the two spectra.

    Recovering A[k] and B[k] from the packed Z needs both Z[k] and its mirror Z[(N-k) mod N], so
    a band of m modes forces 2m-1 rows of the DFT matrix rather than m. That is the scheme's real
    cost and narrowing the band does not reduce it: at m = N/2 the mirror rows are the band's own
    rows and the matrix is as wide as the full transform's.

    What it buys is a matmul half as wide, because the packed operand is K/2 columns rather than
    K. The trade is therefore 2m-1 rows against K/2 columns, and which side wins is a question
    about the shape, not about the arithmetic.

    The packed halves are the left and right column blocks of the input, so [a | b] is the input
    itself and the operand costs nothing to form.

    Real (n, k_cols) with k_cols even -> (re, im), each (n_modes, k_cols).
    """
    n, k_cols = signal.shape
    if k_cols % 2:
        raise ValueError(f"coupled needs an even number of signals, got {k_cols}")
    cols = k_cols // 2

    # Built in Python rather than as a tensor: the matrix cache keys on the row tuple.
    rows = list(range(n_modes)) + [(n - k) % n for k in range(1, n_modes)]
    n_rows = len(rows)

    matrix = stacked_rows(rows, n, -1, dtype, signal.device)
    both = torch.matmul(matrix, signal.to(dtype)).to(ACC)

    cos_a, cos_b = both[:n_rows, :cols], both[:n_rows, cols:]
    sin_a, sin_b = both[n_rows:, :cols], both[n_rows:, cols:]
    z_re, z_im = cos_a - sin_b, sin_a + cos_b          # (cos + i sin) @ (a + i b)

    zr, zi = z_re[:n_modes], z_im[:n_modes]
    mr = torch.cat([z_re[:1], z_re[n_modes:]], dim=0)  # the mirror bins, with DC as its own
    mi = torch.cat([z_im[:1], z_im[n_modes:]], dim=0)
    a_re, a_im = 0.5 * (zr + mr), 0.5 * (zi - mi)
    b_re, b_im = 0.5 * (zi + mi), -0.5 * (zr - mr)
    return (torch.cat([a_re, b_re], dim=1).to(dtype),
            torch.cat([a_im, b_im], dim=1).to(dtype))


def rfft_reduced_low_modes(signal, n_modes, dtype=F):
    """Transform the even and odd samples at half length and recombine: X[k] = E[k] + W_N^k O[k].

    A band of m modes needs E and O at m rows only, so both halves are partial transforms. The
    arithmetic does not improve on a plain partial DFT -- two m x (N/2) matrices hold the same
    m*N entries as one m x N matrix, and the recombination is added on top. Folding the twiddle
    into the odd matrix turns the pair back into exactly the partial DFT's matrix, so the split's
    only contribution at a band is work it then has to undo.

    What is avoidable is how it is issued. A contiguous (n, k_cols) array viewed as
    (n/2, 2*k_cols) already has [even_i | odd_i] side by side in row i, so the operand needs no
    stride-2 gather, and stacking cos over sin gives all four products from one matmul instead of
    four narrow ones. The scheme is therefore a bet on the shape being worth more than the extra
    elementwise pass.

    Real (n, k_cols) with n even -> (re, im), each (n_modes, k_cols).
    """
    n, k_cols = signal.shape
    if n % 2:
        raise ValueError(f"reduced needs an even signal length, got {n}")
    half = n // 2
    operand = signal.reshape(half, 2 * k_cols)         # row i = [even_i | odd_i], no gather

    matrix = stacked_rows(range(n_modes), half, -1, dtype, signal.device)
    both = torch.matmul(matrix, operand.to(dtype)).to(ACC)

    e_re, o_re = both[:n_modes, :k_cols], both[:n_modes, k_cols:]
    e_im, o_im = both[n_modes:, :k_cols], both[n_modes:, k_cols:]
    w_re, w_im = _twiddle(n_modes, n, ACC, signal.device)
    return (e_re + w_re * o_re - w_im * o_im,
            e_im + w_re * o_im + w_im * o_re)


# -- the same two schemes applied to a STAGED full-spectrum transform -------------------------
#
# Above, the packing sits on a direct partial DFT, which is what the layer runs. These two sit
# on the radix-staged transform instead, which is a different machine: the staged plan's cost
# is set by how cleanly N factors into radix-sized stages, so halving the length can remove a
# stage outright rather than merely shrinking a matrix. That is a saving the band versions
# cannot reach, and it is why the two families have to be measured separately.


def rfft_coupled_half(signal, dtype=F):
    """Pair the K signals, run one staged length-N complex transform, separate the two halves.

    Real (n, k_cols) with k_cols even -> (re, im), each (n // 2, k_cols).
    """
    n, k_cols = signal.shape
    if k_cols % 2:
        raise ValueError(f"coupled needs an even number of signals, got {k_cols}")
    half, cols = n // 2, k_cols // 2
    a, b = signal[:, :cols], signal[:, cols:]
    z_re, z_im = fft_staged_complex(a.to(dtype), b.to(dtype), n)

    # Z[(n - k) mod n] by slice reversal, so the mirror costs a copy and not a gather.
    mirror_re = torch.cat([z_re[:1], torch.flip(z_re[1:], (0,))], dim=0)[:half]
    mirror_im = -torch.cat([z_im[:1], torch.flip(z_im[1:], (0,))], dim=0)[:half]
    zr, zi = z_re[:half], z_im[:half]
    a_re, a_im = 0.5 * (zr + mirror_re), 0.5 * (zi + mirror_im)
    b_re, b_im = 0.5 * (zi - mirror_im), -0.5 * (zr - mirror_re)
    return (torch.cat([a_re, b_re], dim=1).to(dtype),
            torch.cat([a_im, b_im], dim=1).to(dtype))


def rfft_reduced_half(signal, dtype=F):
    """Split even and odd, run one staged length-N/2 transform, recombine with the twiddle.

    Real (n, k_cols) with n even -> (re, im), each (n // 2, k_cols).
    """
    n, k_cols = signal.shape
    if n % 2:
        raise ValueError(f"reduced needs an even signal length, got {n}")
    half = n // 2
    pairs = signal.reshape(half, 2, k_cols).to(dtype)
    z_re, z_im = fft_staged_complex(pairs[:, 0, :], pairs[:, 1, :], half)

    mirror_re = torch.cat([z_re[:1], torch.flip(z_re[1:], (0,))], dim=0)
    mirror_im = -torch.cat([z_im[:1], torch.flip(z_im[1:], (0,))], dim=0)
    e_re, e_im = 0.5 * (z_re + mirror_re), 0.5 * (z_im + mirror_im)
    o_re, o_im = 0.5 * (z_im - mirror_im), -0.5 * (z_re - mirror_re)
    w_re, w_im = _twiddle(half, n, ACC, signal.device)
    return ((e_re + w_re * o_re - w_im * o_im).to(dtype),
            (e_im + w_re * o_im + w_im * o_re).to(dtype))
