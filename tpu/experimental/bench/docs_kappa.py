"""Leaf size (kappa = leaf/b) swept at LARGE N, against splitting all the way."""
import csv, os, time
import jax, jax.numpy as jnp, numpy as np
from ...fft.core import F  # noqa: F401
from ...fft.rfft1d import rfft_half
from ..fft.autoplan import rfft_two_factor

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_kappa")
REPS = 20

def run(tag, fn, x):
    j = jax.jit(fn); jax.block_until_ready(j(x)); ts=[]
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0=time.perf_counter(); jax.block_until_ready(j(x)); ts.append(time.perf_counter()-t0)
    try:
        c=jax.jit(fn).lower(x).compile().cost_analysis()
        if isinstance(c,list): c=c[0]
        return sorted(ts)[len(ts)//2]*1e6, float(c.get("bytes accessed",0.0)), float(c.get("flops",0.0))
    except Exception:
        return sorted(ts)[len(ts)//2]*1e6, float("nan"), float("nan")

def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    K, rows = 256, []
    print(f"\n  {'N':<8}{'plan':<22}{'kappa':>6}{'wall us':>10}{'HBM MB':>10}{'GFLOP':>9}{'err':>10}", flush=True)
    for n in (2048, 4096, 8192, 16384):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        truth = np.fft.fft(np.asarray(x, np.float64), axis=0)[:n//2]
        cands = [("split all the way", None)] + [(f"leaf {l}", l) for l in (128,256,512,1024) if n % l == 0 and l <= n]
        for name, leaf in cands:
            fn = (lambda z: rfft_half(z, F)) if leaf is None else (lambda z, l=leaf: rfft_two_factor(z, l))
            try:
                w, hbm, fl = run(f"N{n}_{'split' if leaf is None else leaf}", fn, x)
                o = jax.jit(fn)(x)
                got = np.asarray(o[0], np.float64) + 1j*np.asarray(o[1], np.float64)
                err = float(np.max(np.abs(got-truth))/np.max(np.abs(truth)))
            except Exception as e:
                print(f"  {n:<8}{name:<22}FAILED {type(e).__name__}: {str(e)[:35]}", flush=True); continue
            kap = "-" if leaf is None else f"{leaf//128}"
            print(f"  {n:<8}{name:<22}{kap:>6}{w:>10.1f}{hbm/1e6:>10.2f}{fl/1e9:>9.3f}{err:>10.1e}", flush=True)
            rows.append({"n":n,"plan":name,"leaf":leaf or 0,"kappa":kap,"wall_us":w,
                         "hbm_mb":hbm/1e6,"gflop":fl/1e9,"rel_err":err,
                         "trace":f"N{n}_{'split' if leaf is None else leaf}"})
        print(flush=True)
    with open(os.environ.get("DOCS_KAPPA_CSV", "/tmp/docs_kappa.csv"),"w",newline="") as fh:
        wr=csv.DictWriter(fh,fieldnames=sorted({k for r in rows for k in r})); wr.writeheader(); wr.writerows(rows)
    print("KAPPA_DONE", flush=True)

if __name__ == "__main__":
    main()
