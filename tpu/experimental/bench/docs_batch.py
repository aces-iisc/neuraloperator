"""Sweeps inference batch size: the mix is a per-mode matmul whose N axis is the batch."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno import layers, model, spectral

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_batch")
CSV = os.environ.get("DOCS_BATCH_CSV", "/tmp/docs_batch.csv")
FIELDS = ["case", "what", "batch", "wall_us_total", "us_per_sample", "status", "trace"]
REPS = 20
BATCHES = (8, 16, 32, 64, 128, 256)
CASES = [("Darcy-64", 24, 16, 64), ("NS-128", 64, 64, 128)]


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def run(tag, fn, *a):
    j = jax.jit(fn)
    jax.block_until_ready(j(*a))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(*a)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]*1e6


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, C, nm, g in CASES:
        P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, channels=C, n_modes=nm, n_layers=4)
        w = layers._init_spectral_weight(jax.random.PRNGKey(0), C, C, nm)
        print(f"\n### {label}  channels={C} n_modes={nm} grid={g}x{g}", flush=True)
        print(f"  {'what':<14}{'batch':>6}{'total us':>11}{'us/sample':>12}", flush=True)
        for what in ("spectral conv", "full forward"):
            for B in BATCHES:
                tag = f"{label}_{what.replace(' ','')}_{B}"
                try:
                    if what == "spectral conv":
                        x = jnp.asarray(np.random.default_rng(0).standard_normal(
                            (B, C, g, g)).astype(np.float32))
                        fn = lambda z: spectral.spectral_conv(w, z, nm)
                    else:
                        x = jnp.asarray(np.random.default_rng(0).standard_normal(
                            (B, 1, g, g)).astype(np.float32))
                        fn = lambda z: model.forward(P, z, nm)
                    t = run(tag, fn, x)
                    print(f"  {what:<14}{B:>6}{t:>11.1f}{t/B:>12.3f}", flush=True)
                    rows.append({"case": label, "what": what, "batch": B, "wall_us_total": t,
                                 "us_per_sample": t/B, "status": "ok", "trace": tag})
                except Exception as exc:
                    why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    print(f"  {what:<14}{B:>6}{why:>11}", flush=True)
                    rows.append({"case": label, "what": what, "batch": B,
                                 "wall_us_total": float("nan"), "us_per_sample": float("nan"),
                                 "status": why, "trace": ""})
                save(rows)
            print(flush=True)
    save(rows)
    print("BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
