"""Fused Pallas spectral conv against the XLA one, with HBM traffic from the cost model."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno import layers, spectral
from ..fno.pallas_fused import fused_spectral_conv

CSV = os.environ.get("DOCS_FUSED_CSV", "/tmp/docs_fused.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_fused")
FIELDS = ["case", "variant", "block", "wall_us", "hbm_mb", "rel_diff", "status", "trace"]
REPS = 20
CASES = [("Darcy-32", 24, 16, 32, 8), ("NS-128", 64, 64, 128, 8)]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def measure(tag, fn, x):
    j = jax.jit(fn); jax.block_until_ready(j(x))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(x)); ts.append(time.perf_counter()-t0)
    try:
        c = j.lower(x).compile().cost_analysis()
        if isinstance(c, list): c = c[0]
        h = float(c.get("bytes accessed", 0.0))/1e6
    except Exception:
        h = float("nan")
    return sorted(ts)[len(ts)//2]*1e6, h


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, C, nm, g, B in CASES:
        w = layers._init_spectral_weight(jax.random.PRNGKey(0), C, C, nm)
        x = jnp.asarray(np.random.default_rng(0).standard_normal((B, C, g, g)).astype(np.float32))
        ref = np.asarray(spectral.spectral_conv(w, x, nm), np.float64)
        den = np.max(np.abs(ref))
        wt_mb = (w["w_re"].size + w["w_im"].size)*4/1e6
        print(f"\n### {label}  B={B} C={C} modes={nm} grid={g}  weights {wt_mb:.2f} MB", flush=True)
        print(f"  {'variant':<18}{'block':>6}{'wall us':>10}{'HBM MB':>10}{'rel diff':>11}", flush=True)
        t, h = measure(f"{label}_xla", lambda z: spectral.spectral_conv(w, z, nm), x)
        print(f"  {'xla (current)':<18}{'-':>6}{t:>10.1f}{h:>10.2f}{0.0:>11.1e}", flush=True)
        rows.append({"case": label, "variant": "xla", "block": 0, "wall_us": t, "hbm_mb": h,
                     "rel_diff": 0.0, "status": "ok", "trace": f"{label}_xla"}); save(rows)
        for blk in (1, 2, 4, 8):
            if B % blk: continue
            tag = f"{label}_fused{blk}"
            try:
                fn = lambda z, b=blk: fused_spectral_conv(w, z, nm, block=b, interpret=False,
                                                          vmem=64*1024*1024)
                t, h = measure(tag, fn, x)
                d = float(np.max(np.abs(np.asarray(jax.jit(fn)(x), np.float64)-ref))/den)
                print(f"  {'pallas fused':<18}{blk:>6}{t:>10.1f}{h:>10.2f}{d:>11.1e}", flush=True)
                rows.append({"case": label, "variant": "fused", "block": blk, "wall_us": t,
                             "hbm_mb": h, "rel_diff": d, "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"  {'pallas fused':<18}{blk:>6}{why:>10}   {str(exc)[:46]}", flush=True)
                rows.append({"case": label, "variant": "fused", "block": blk,
                             "wall_us": float("nan"), "hbm_mb": float("nan"),
                             "rel_diff": float("nan"), "status": why, "trace": ""})
            save(rows)
    save(rows); print("\nFUSED_DONE", flush=True)


if __name__ == "__main__":
    main()
