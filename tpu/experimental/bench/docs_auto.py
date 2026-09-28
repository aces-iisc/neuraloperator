"""
The auto-planned real FFT against reim, coupled and reduced, swept over N.
Device time comes from the xprof traces; wall-clock has a ~180us floor at these sizes.

    PYTHONPATH=~/nop ~/nopenv/bin/python -m tpu.experimental.bench.docs_auto
"""
import csv, os, time
import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F  # noqa: F401  (precision HIGHEST)
from ...fft.rfft1d import rfft_half
from ..fft.autoplan import choose, rfft_auto, rfft_plan, tile_cost
from ..fft.tricks import rfft_coupled_reim, rfft_reduced_reim

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_auto")
REPS = 20


def run(tag, fn, x):
    j = jax.jit(fn)
    jax.block_until_ready(j(x))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(x)); ts.append(time.perf_counter()-t0)
    wall = sorted(ts)[len(ts)//2]*1e6
    try:
        c = jax.jit(fn).lower(x).compile().cost_analysis()
        if isinstance(c, list): c = c[0]
        return wall, float(c.get("bytes accessed",0.0)), float(c.get("flops",0.0))
    except Exception:
        return wall, float("nan"), float("nan")


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    K = 256
    rows = []
    print(f"\n  {'N':<8}{'engine':<14}{'plan':<10}{'tiles':>8}{'wall us':>10}{'HBM MB':>10}{'rel err':>10}", flush=True)
    for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        truth = np.fft.fft(np.asarray(x, np.float64), axis=0)[:n//2]
        L, m = choose(n, K)
        engines = [("auto", lambda z: rfft_auto(z), f"L={L},m={m}", tile_cost(n, K, L, m)),
                   ("reim", lambda z: rfft_half(z, F), "", 0),
                   ("coupled", lambda z: rfft_coupled_reim(z, 0.5, F), "", 0),
                   ("reduced", lambda z: rfft_reduced_reim(z, 0.5, F), "", 0)]
        for name, fn, planstr, tiles in engines:
            try:
                wall, hbm, fl = run(f"N{n}_{name}", fn, x)
                o = jax.jit(fn)(x)
                got = np.asarray(o[0], np.float64) + 1j*np.asarray(o[1], np.float64)
                err = float(np.max(np.abs(got - truth))/np.max(np.abs(truth)))
            except Exception as exc:
                print(f"  {n:<8}{name:<14}FAILED {type(exc).__name__}: {str(exc)[:40]}", flush=True)
                continue
            print(f"  {n:<8}{name:<14}{planstr:<10}{tiles if tiles else '':>8}{wall:>10.1f}"
                  f"{hbm/1e6:>10.2f}{err:>10.1e}", flush=True)
            rows.append({"n": n, "engine": name, "plan": planstr, "tiles": tiles,
                         "wall_us": wall, "hbm_mb": hbm/1e6, "gflop": fl/1e9,
                         "rel_err": err, "trace": f"N{n}_{name}"})
        print(flush=True)
    with open("/tmp/docs_auto.csv","w",newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print("AUTO_DONE", flush=True)


if __name__ == "__main__":
    main()
