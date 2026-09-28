"""
VMEM-resident 2D real rfft (full) as one Pallas kernel.

Real (N1,N2,K) -> HALF spectrum (yr,yi), each (N1//2,N2,K). One kernel per K-tile does the whole
2D transform on-chip: axis-0 staged real rfft, axis-1 direct complex DFT (contract middle axis,
no transpose). Kernel form of fft.rfft2d.rfft2_full; for the FNO's kept modes use rfft2_low_modes.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...fft.core import F, rfft_staged as _fft_reim_real
from ...fft.rfft2d import _dft_axis1

_PARALLEL = pltpu.CompilerParams(dimension_semantics=("parallel",))


def _rfft2_body(xr, N1, N2, dt):
    """2D rfft of one (N1,N2,K_TILE) real block in VMEM: real rfft on axis-0, complex DFT axis-1."""
    K = xr.shape[2]
    Ar, Ai = _fft_reim_real(xr.reshape(N1, N2 * K), N1, half=True, dtype=dt)
    R = N1 // 2
    Ar, Ai = Ar.reshape(R, N2, K), Ai.reshape(R, N2, K)
    return _dft_axis1(Ar, Ai, N2, dtype=dt)


def _kernel2d(xr_ref, yr_ref, yi_ref, *, N1, N2, dt):
    yr_ref[...], yi_ref[...] = _rfft2_body(xr_ref[...], N1, N2, dt)


def pallas_rfft2(xr, K_TILE=128, interpret=True, dt=F):
    """Real (N1,N2,K) -> HALF spectrum (yr,yi), each (N1//2,N2,K). Grid over K; each tile does the
    full 2D transform in VMEM. dt=BF gives the bf16 variant."""
    N1, N2, K = xr.shape
    R = N1 // 2
    return pl.pallas_call(
        partial(_kernel2d, N1=N1, N2=N2, dt=dt),
        grid=(K // K_TILE,),
        in_specs=[pl.BlockSpec((N1, N2, K_TILE), lambda k: (0, 0, k))],
        out_specs=[pl.BlockSpec((R, N2, K_TILE), lambda k: (0, 0, k))] * 2,
        out_shape=[jax.ShapeDtypeStruct((R, N2, K), dt)] * 2,
        compiler_params=_PARALLEL,
        interpret=interpret,
    )(xr.astype(dt))
