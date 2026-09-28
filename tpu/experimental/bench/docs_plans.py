"""Three factor orders for the same 1-D transform, measured."""
import csv, os, time
import jax, jax.numpy as jnp, numpy as np
from ...fft.core import F  # noqa: F401
from ...fft.rfft1d import rfft_half
from ..fft.autoplan import rfft_auto, rfft_leaf_first

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_plans")
REPS = 20

def run(tag, fn, x):
    j = jax.jit(fn); jax.block_until_ready(j(x)); ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(x)); ts.append(time.perf_counter()-t0)
    try:
        c = jax.jit(fn).lower(x).compile().cost_analysis()
        if isinstance(c, list): c = c[0]
        return sorted(ts)[len(ts)//2]*1e6, float(c.get("bytes accessed",0.0)), float(c.get("flops",0.0))
    except Exception:
        return sorted(ts)[len(ts)//2]*1e6, float("nan"), float("nan")

def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    K, rows = 256, []
    print(f"\n  {'N':<8}{'plan':<26}{'wall us':>10}{'HBM MB':>10}{'GFLOP':>9}", flush=True)
    for n in (256, 512, 1024, 2048):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        for name, fn in (("split 128 then leaf (now)", lambda z: rfft_half(z, F)),
                         ("leaf-first (mixed radix)", rfft_leaf_first),
                         ("direct / auto", rfft_auto)):
            try:
                w, hbm, fl = run(f"N{n}_{name.split()[0]}", fn, x)
            except Exception as e:
                print(f"  {n:<8}{name:<26}FAILED {type(e).__name__}", flush=True); continue
            print(f"  {n:<8}{name:<26}{w:>10.1f}{hbm/1e6:>10.2f}{fl/1e9:>9.3f}", flush=True)
            rows.append({"n": n, "plan": name, "wall_us": w, "hbm_mb": hbm/1e6, "gflop": fl/1e9,
                         "trace": f"N{n}_{name.split()[0]}"})
        print(flush=True)
    with open("/tmp/docs_plans.csv","w",newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r})); wr.writeheader(); wr.writerows(rows)
    print("PLANS_DONE", flush=True)

if __name__ == "__main__":
    main()
