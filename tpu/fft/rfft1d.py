import jax.numpy as jnp

from .core import F, dft_matrix_reim, rfft_staged


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


def rfft_low_modes(signal, n_modes, dtype=F):
    """Compute ONLY the lowest `n_modes` bins of a REAL (n, batch) input, as one direct
    (n_modes x n) real/imag DFT matmul — no staging, no twiddle, no transpose, and a single HBM
    pass (read the signal once, write the small result).  Returns (re, im), each (n_modes, batch).

    Worth it only for n_modes << n: the DFT matrix is O(n_modes*n), so keeping half the spectrum
    would cost O(n^2).  Trades cheap MXU flops for fewer HBM passes than the staged radix rfft,
    which is the winning trade while the transform is memory-bound.  1-D analogue of
    rfft2d.rfft2_low_modes."""
    n = signal.shape[0]
    angle = -2.0 * jnp.pi * jnp.outer(jnp.arange(n_modes), jnp.arange(n)).astype(F) / n
    dft_cos, dft_sin = jnp.cos(angle).astype(dtype), jnp.sin(angle).astype(dtype)
    x = signal.astype(dtype)
    return (jnp.matmul(dft_cos, x, preferred_element_type=jnp.float32),
            jnp.matmul(dft_sin, x, preferred_element_type=jnp.float32))


def rfft_direct(signal, half=True, dtype=F):
    """One n x n DFT, no staging. O(n^2) arithmetic but every MXU tile is full, which wins at
    small n where staging leaves a starved leaf. Real (n, batch) -> (re, im)."""
    n = signal.shape[0]
    cos, sin, _ = dft_matrix_reim(n, dtype)
    x = signal.astype(dtype)
    re = jnp.matmul(cos, x, preferred_element_type=jnp.float32)
    im = jnp.matmul(sin, x, preferred_element_type=jnp.float32)
    return (re[:n // 2], im[:n // 2]) if half else (re, im)
