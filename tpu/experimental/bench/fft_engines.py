"""
bench_fft_engines.py — run ON the v5e.  Re-assessment of the NO-complex64 FFT engines, 1D + 2D,
at large N, ratio 0.5 (half spectrum).  Reports median blocked wall-clock us/run (K=256 batch) and
us/sample, plus a complex64-count column (must be 0 for every engine here — that's the point).

Engines (all real/imag, no c64 custom-call):
  1D : reim (direct half rfft) | coupled (reim) | reduced (reim) | pallas (full FFT resident in VMEM)
  2D : reim (full real rfft)   | partial (rfft2_partial, low modes as GEMM) | pallas (full 2D in VMEM)

Pallas computes the WHOLE transform in VMEM (NOT partial) -> it OOMs/hangs at large N, so it is
run only up to a safe cap and marked 'skip' above it (never at a size that wedges the VM).
"""
import time
import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F  # noqa: F401  (import sets global highest matmul precision)
from ...fft.rfft1d import rfft_half, rfft_low_modes as rfft1d_partial
from ...fft.rfft2d import (rfft2_full as jax_rfft2, rfft2_full_staged_cols as jax_rfft2_fast,
                           rfft2_low_modes as rfft2_partial)
from ..fft.pallas1d import pallas_half
from ..fft.pallas2d import pallas_rfft2
from ..fft.tricks import rfft_coupled_reim, rfft_reduced_reim

print("jax", jax.__version__, "|", jax.devices(), flush=True)
K, KT, ITP = 256, 128, False


def times(fn, x, reps=15):
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def c64_count(fn, x):
    try:
        return jax.jit(fn).lower(x).compile().as_text().count("c64") + \
               jax.jit(fn).lower(x).compile().as_text().count("complex64")
    except Exception:
        return -1


ENG1D = {"reim":   lambda x: rfft_half(x, F),            # full half-spectrum (N/2 modes)
         "coupled": lambda x: rfft_coupled_reim(x, 0.5, F),
         "reduced": lambda x: rfft_reduced_reim(x, 0.5, F)}
ENG2D = {"reim":      lambda x: jax_rfft2(x, F),              # axis-1 = direct O(N2^2) DFT
         "reim-fast": lambda x: jax_rfft2_fast(x, F),         # axis-1 = STAGED FFT O(N2 log N2)
         "partial":   lambda x: rfft2_partial(x, 0.5, F),
         "pallas":    lambda x: pallas_rfft2(x, KT, ITP, F)}  # full 2D in VMEM -> hangs >=128

N1D, PALLAS_1D_MAX = [1024, 4096, 16384, 65536, 262144], 4096
N2D, PALLAS_2D_MAX = [32, 64, 128, 256, 512], 64

print(f"\n### complex64 op-count (must be 0 — no-c64 property) ###", flush=True)
xc = jnp.asarray(np.random.randn(1024, K).astype(np.float32))
for nm, fn in ENG1D.items():
    print(f"  1D {nm:8s}: c64_ops={c64_count(fn, xc)}", flush=True)
xc2 = jnp.asarray(np.random.randn(64, 64, K).astype(np.float32))
for nm, fn in ENG2D.items():
    print(f"  2D {nm:8s}: c64_ops={c64_count(fn, xc2)}", flush=True)

print(f"\n### 1D rfft (no c64), median us/run  [us/sample=/{K}] @ K={K} ###", flush=True)
for N in N1D:
    x = jnp.asarray(np.random.randn(N, K).astype(np.float32))
    cells = []
    for nm, fn in ENG1D.items():
        if nm == "pallas" and N > PALLAS_1D_MAX:
            cells.append(f"{nm}=skip(OOM)"); continue
        try:
            us = times(jax.jit(fn), x) * 1e6
            cells.append(f"{nm}={us:.1f}({us/K:.2f})")
        except Exception as e:
            cells.append(f"{nm}=FAIL:{type(e).__name__}")
    print(f"  N={N:>8}: " + "  ".join(cells), flush=True)

print(f"\n### 2D rfft (no c64), median us/run  [us/sample=/{K}] @ K={K} ###", flush=True)
for N in N2D:
    x = jnp.asarray(np.random.randn(N, N, K).astype(np.float32))
    cells = []
    for nm, fn in ENG2D.items():
        if nm == "pallas" and N > PALLAS_2D_MAX:
            cells.append(f"{nm}=skip(hang)"); continue
        try:
            us = times(jax.jit(fn), x) * 1e6
            cells.append(f"{nm}={us:.1f}({us/K:.2f})")
        except Exception as e:
            cells.append(f"{nm}=FAIL:{type(e).__name__}")
    print(f"  {N}x{N}: " + "  ".join(cells), flush=True)
print("FFTBENCH_DONE", flush=True)
