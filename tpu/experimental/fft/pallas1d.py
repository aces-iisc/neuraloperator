"""
VMEM-resident radix-B REAL-input 1D FFT as a single Pallas TPU kernel.

Both public kernels take REAL input (N, K); they differ only in output length:
    pallas_full  -> all N bins        pallas_half  -> first N//2 bins (Hermitian rfft)
The whole transform runs in ONE kernel with the working set resident in VMEM, so the only HBM
traffic is: read the input once, write the output once (intermediates never leave VMEM).
pallas_reduced is the `reduced` variant with a self-made (Mosaic-lowerable) row reversal.
Numerics come from fft/core.py (imported here under their old short aliases).
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...fft.core import (
    BYTES_PER_COMPONENT as _BYTES,
    RADIX as B,
    F,
    complex_matmul as _karatsuba,
    dft_matrix_reim as _CB_reim,
    fft_staged_complex as _fft_c,
    reverse_rows as _rev,
    rfft_staged as _fft_reim_real,
)

# K-grid is embarrassingly parallel (each K-tile is an independent transform); telling Mosaic
# so lets it pipeline the tiles — overlap tile k+1's HBM read with tile k's compute.
_PARALLEL = pltpu.CompilerParams(dimension_semantics=("parallel",))


def _kernel(xr_ref, yr_ref, yi_ref, *, N, half, dt):
    """Kernel body: whole FFT of one K_TILE-column block, entirely in VMEM."""
    yr_ref[...], yi_ref[...] = _fft_reim_real(xr_ref[...], N, half, dt)


def _pallas(xr, K_TILE, interpret, half, dt):
    """Shared pallas_call: grid over K; output rows = N//2 (half) or N (full), dtype dt."""
    N, K = xr.shape
    rows = N // 2 if half else N
    return pl.pallas_call(
        partial(_kernel, N=N, half=half, dt=dt),
        grid=(K // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((rows, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((rows, K), dt)] * 2,
        compiler_params=_PARALLEL,
        interpret=interpret,
    )(xr)


def pallas_full(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N, K) -> FULL spectrum (yr, yi), each (N, K)  — all N bins.
    dt=BF gives the bf16 'complex32' variant (half the bytes, ~5e-3 error)."""
    return _pallas(xr, K_TILE, interpret, half=False, dt=dt)


def pallas_half(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N, K) -> HALF spectrum (yr, yi), each (N//2, K)  — Hermitian rfft.
    dt=BF gives the bf16 'complex32' variant (half the bytes, ~5e-3 error)."""
    return _pallas(xr, K_TILE, interpret, half=True, dt=dt)


def pallas_hbm_bytes(N, K, half=False, dt=F):
    """Analytical HBM traffic (VMEM-resident): read input once + write output once.
    Intermediates never touch HBM by design.  half writes only N//2 rows.
    dt sets bytes/component: f32 -> 4 (complex64), bf16 -> 2 ('complex32')."""
    dtb = _BYTES[dt]
    rows = N // 2 if half else N
    return N * K * dtb + 2 * rows * K * dtb  # read xr + write (yr, yi)


# ── Trick B (even/odd) with self-made row reversal ───────────────────────────
def _reduced(xr, N):
    """Even/odd -> half-length FFT -> self-made mirror -> twist.  Returns first N//2 bins."""
    h, batch = N // 2, xr.shape[1]
    x2 = xr.reshape(h, 2, batch)
    Zr, Zi = _fft_c(x2[:, 0, :], x2[:, 1, :], h)  # HALF-length FFT of (even + i*odd)
    Mr = pltpu.roll(_rev(Zr), 1, axis=0)  # mirror M[k]=conj(Z[(h-k)%h]) = roll(rev(Z),1)
    Mi = -pltpu.roll(_rev(Zi), 1, axis=0)  # conj -> negate imag
    Er, Ei = 0.5 * (Zr + Mr), 0.5 * (Zi + Mi)  # even sub-DFT E = 0.5(Z+M)
    Dr, Di = Zr - Mr, Zi - Mi
    Or_, Oi = 0.5 * Di, -0.5 * Dr  # odd sub-DFT O = -0.5j(Z-M)
    k = jnp.arange(h)
    twr = jnp.cos(-2 * jnp.pi * k / N).astype(F)[:, None]
    twi = jnp.sin(-2 * jnp.pi * k / N).astype(F)[:, None]
    return Er + (twr * Or_ - twi * Oi), Ei + (twr * Oi + twi * Or_)  # X[k]=E[k]+W_N^k O[k]


def _reduced_kernel(xr_ref, yr_ref, yi_ref, *, N):
    yr_ref[...], yi_ref[...] = _reduced(xr_ref[...], N)


def pallas_reduced(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N, K) -> HALF spectrum via Trick B + self-made reversal.  Correct and lowers, but
    the O(N^2) mirror makes it slower than pallas_half.  (dt accepted for a uniform interface;
    the `reduced` path runs in f32.)"""
    xr = xr.astype(F)
    N, K = xr.shape
    h = N // 2
    return pl.pallas_call(
        partial(_reduced_kernel, N=N),
        grid=(K // K_TILE,),
        in_specs=[pl.BlockSpec((N, K_TILE), lambda k: (0, k))],
        out_specs=[pl.BlockSpec((h, K_TILE), lambda k: (0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((h, K), F)] * 2,
        compiler_params=_PARALLEL,
        interpret=interpret,
    )(xr)
