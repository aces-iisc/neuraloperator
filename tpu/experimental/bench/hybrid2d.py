"""
bench_hybrid2d.py — run ON the v5e.  Tests the coupled+partial HYBRID (full FFT on axis-0 cols,
partial low-mode DFT on axis-1 rows) vs the all-partial and full-reim 2D transforms, at real
dataset (N, m) with realistic field batches K = batch x width.
"""
import time
import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F  # noqa: F401  (import sets global highest matmul precision)
from ...fft.rfft2d import rfft2_full as jax_rfft2, rfft2_low_modes as rfft2_partial
from ..fft.rfft2_hybrid import rfft2_hybrid

print("jax", jax.__version__, "|", jax.devices(), flush=True)


def bench(fn, x, reps=20, warm=3):
    for _ in range(warm):
        jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(fn(x)); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e6


CFG = [("NS", 128, 16, 1024), ("Darcy", 32, 12, 256)]   # (label, N, m, K = batch*width fields)
for label, N, m, K in CFG:
    x = jnp.asarray(np.random.randn(N, N, K).astype(np.float32))
    eng = {"partial(both)":       lambda x, m=m, N=N: rfft2_partial(x, m / N, F),
           "hyb reim+partial":    lambda x, m=m: rfft2_hybrid(x, m, m, "reim", "partial", F),
           "hyb reim+staged-rows": lambda x, m=m: rfft2_hybrid(x, m, m, "reim", "staged", F),
           "hybrid-coupled":       lambda x, m=m: rfft2_hybrid(x, m, m, "coupled", "partial", F),
           "hybrid-reduced":       lambda x, m=m: rfft2_hybrid(x, m, m, "reduced", "partial", F),
           "reim(full both)":     lambda x: jax_rfft2(x, F)}
    print(f"\n### {label} 2D: N={N}^2, m={m}, K={K} fields (median us/run) ###", flush=True)
    for name, fn in eng.items():
        try:
            print(f"  {name:16s}: {bench(jax.jit(fn), x):9.1f} us", flush=True)
        except Exception as e:
            print(f"  {name:16s}: FAIL {type(e).__name__}: {str(e).splitlines()[0][:60]}", flush=True)
print("HYBRID_BENCH_DONE", flush=True)
