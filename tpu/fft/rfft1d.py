import jax.numpy as jnp

from .core import ACC, F, rfft_staged


def rfft_full(signal):
    """Full n-bin FFT of a REAL (n, batch) input, carried entirely in real/imag — no complex64.
    Takes the optimized real path (first stage is 2 matmuls, not the 3-matmul Karatsuba of the
    general complex core), so it exploits the input being real.  Returns (re, im)."""
    return rfft_staged(signal.astype(F), signal.shape[0], half=False)


def rfft_half(signal, dtype=F):
    """The n//2 non-redundant bins of a REAL (n, batch) input.  `dtype` picks precision:
    F is the complex64 equivalent, BF the bf16 "complex32" one that measured fastest at every
    n >= 1024 (results/REPORT.md §1).  Returns (re, im)."""
    return rfft_staged(signal, signal.shape[0], half=True, dtype=dtype)


def rfft_rows(signal, rows, dtype=F):
    """The DFT bins listed in `rows` of a REAL (n, batch) input, carried as real/imag.

    `rows` says WHICH output frequencies to evaluate, and nothing outside it is ever computed:
    the matrix is (len(rows), n), not (n, n).  Output frequencies are independent of each other,
    so this is exact, not an approximation -- a restricted Vandermonde, not a truncated FFT.
    The half-spectrum and low-mode cases are therefore the same code with a different row set,
    and neither slices a larger result.

    cos and sin hit the same operand, so they go as ONE matmul with the two matrices stacked
    along the output axis; the MXU sees twice the rows per call.  Returns (re, im), each
    (len(rows), batch).

    A staged radix rfft cannot do this -- its butterfly tree shares subexpressions across all n
    outputs, so it computes the whole spectrum and discards the rest.  That is why rfft_staged
    slices and this does not.
    """
    n = signal.shape[0]
    k = jnp.asarray(rows)[:, None]
    angle = (-2.0 * jnp.pi / n) * (k * jnp.arange(n)[None, :]).astype(F)
    stacked = jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=0).astype(dtype)
    both = jnp.matmul(stacked, signal.astype(dtype), preferred_element_type=ACC)
    n_rows = k.shape[0]
    return both[:n_rows], both[n_rows:]


def rfft_low_modes(signal, n_modes, dtype=F):
    """The lowest `n_modes` bins only, as one direct (2*n_modes x n) real/imag matmul -- no
    staging, no twiddle, no transpose, and a single HBM pass.

    Worth it while n_modes << n: the matrix is O(n_modes*n), so keeping half the spectrum would
    cost O(n^2).  Trades cheap MXU flops for fewer HBM passes than the staged radix rfft, which
    is the winning trade while the transform is memory-bound.  1-D analogue of
    rfft2d.rfft2_low_modes."""
    return rfft_rows(signal, jnp.arange(n_modes), dtype)


def rfft_direct(signal, half=True, dtype=F):
    """One direct DFT, no staging.  O(n^2) arithmetic but every MXU tile is full, which wins at
    small n where staging leaves a starved leaf.  Real (n, batch) -> (re, im).

    `half` picks the row set rather than trimming a full result, so the non-redundant half costs
    half the matmul instead of all of it."""
    n = signal.shape[0]
    return rfft_rows(signal, jnp.arange(n // 2 if half else n), dtype)
