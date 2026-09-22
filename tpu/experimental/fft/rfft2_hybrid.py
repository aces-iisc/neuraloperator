"""
experimental/fft/rfft2_hybrid.py — the low-mode 2-D rfft variants that were tried against
fft.rfft2d.rfft2_partial.  They do a FULL FFT on axis-0 (or a packing trick) and only then keep the
low modes, so they move more bytes than the straight partial DFT.  Measured in
results/fft_engines_reassess.md; kept for the record, not used by the main tree.
"""
import jax.numpy as jnp

from ...fft.core import (ACC, F, complex_matmul as _karatsuba,
                         fft_staged_complex as _fft_c, rfft_staged as _fft_reim_real)
from .tricks import rfft_coupled_reim, rfft_reduced_reim


def rfft2_hybrid(xr, m1, m2, axis0="reim", rows="partial", dt=F):
    """HYBRID low-mode 2D rfft: FULL real FFT on axis-0 (all cols) then keep the low m1; then rows:
      rows='partial' -> PARTIAL direct DFT on axis-1 (skinny (m2 x N2), only the low m2) [the winner]
      rows='staged'  -> full STAGED FFT on axis-1 (transpose, _fft_c) then TRUNCATE to m2 (computes
                        all modes then slices — the 'truncated reim rows' variant to test)
    axis0='reim' (staged rfft) or 'coupled'.  (N1,N2,K) -> (yr,yi), each (m1,m2,K)."""
    N1, N2, K = xr.shape
    x2 = xr.reshape(N1, N2 * K)
    if axis0 == "coupled":
        Ar, Ai = rfft_coupled_reim(x2, m1 / N1, dt)               # full real rfft (coupled), m1 modes
    elif axis0 == "reduced":
        Ar, Ai = rfft_reduced_reim(x2, m1 / N1, dt)               # full real rfft (reduced), m1 modes
    else:
        Ar, Ai = _fft_reim_real(x2, N1, half=True, dtype=dt)        # full staged real rfft
    Ar, Ai = Ar[:m1].reshape(m1, N2, K), Ai[:m1].reshape(m1, N2, K)   # keep low m1 col-modes
    if rows == "staged":                                          # staged FFT on axis-1 + truncate
        Ar2 = Ar.transpose(1, 0, 2).reshape(N2, m1 * K)          # N2 -> front (needs transpose)
        Ai2 = Ai.transpose(1, 0, 2).reshape(N2, m1 * K)
        Yr, Yi = _fft_c(Ar2.astype(dt), Ai2.astype(dt), N2)      # full complex FFT (all N2 modes)
        Yr = Yr.reshape(N2, m1, K).transpose(1, 0, 2)[:, :m2]    # back, TRUNCATE to low m2
        Yi = Yi.reshape(N2, m1, K).transpose(1, 0, 2)[:, :m2]
        return Yr, Yi                                            # (m1, m2, K)
    a2 = -2 * jnp.pi * jnp.outer(jnp.arange(m2), jnp.arange(N2)) / N2  # axis-1 PARTIAL DFT (m2 x N2)
    Cr2, Ci2 = jnp.cos(a2).astype(dt), jnp.sin(a2).astype(dt)
    mm = lambda A, x: jnp.einsum('on,rnk->rok', A, x, preferred_element_type=ACC)
    return _karatsuba(mm, Cr2, Ci2, (Cr2 + Ci2).astype(dt), Ar, Ai)  # (m1, m2, K)
