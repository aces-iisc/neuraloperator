"""
One spectral convolution, timed ONLY at configurations the repository actually runs.

  Darcy  32^2   n_modes=(16,16) channels=24 batch=16   darcy_config + FNO_Small2d
  NS     128^2  n_modes=(64,64) channels=64 batch=8    navier_stokes_config + FNO_Medium2d
  NS     1024^2 n_modes=(64,64) channels=64 batch=4    default_config test_resolutions[1]

Nothing here is a synthetic sweep point: n_modes and channels come from the shipped model presets,
the grids from the configured train/test resolutions, and the batch sizes from test_batch_sizes.
Note that 1024^2 is a TEST resolution -- the model trains at 128^2 and is evaluated there zero
shot, reusing the same n_modes because the band is fixed in wavenumbers.

    PYTHONPATH=~/nop ~/nopenv/bin/python -m tpu.experimental.bench.docs_fno_real
"""
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import BF, F  # noqa: F401  (sets precision HIGHEST globally)
from ...fno.layers import _init_spectral_weight
from ...fno.spectral import spectral_conv
from ..fno import reference
from ..fno.spectral_variants import spectral_2corner_herm, spectral_2corner_hybrid

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_real")
REPS = 20

# (label, grid, n_modes, channels, batch, provenance)
CONFIGS = [
    ("Darcy 32^2", 32, 16, 24, 16, "darcy_config, FNO_Small2d"),
    ("NS 128^2", 128, 64, 64, 8, "navier_stokes_config, FNO_Medium2d"),
    ("NS 1024^2", 1024, 64, 64, 4, "default_config test res, FNO_Medium2d"),
]


def run_one(tag, fn, *args):
    run = jax.jit(fn)
    jax.block_until_ready(run(*args))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(run(*args))
            ts.append(time.perf_counter() - t0)
    wall = sorted(ts)[len(ts) // 2] * 1e6
    try:
        cost = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(cost, list):
            cost = cost[0]
        hbm = float(cost.get("bytes accessed", 0.0))
    except Exception:
        hbm = float("nan")
    return wall, hbm, os.path.join(LOGDIR, tag)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("matmul precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []

    for label, grid, m, channels, batch, prov in CONFIGS:
        w = _init_spectral_weight(jax.random.PRNGKey(0), channels, channels, m)
        rng = np.random.default_rng(0)
        x = jnp.asarray(rng.standard_normal((batch, channels, grid, grid)).astype(np.float32))
        print(f"\n### {label}  n_modes={m} channels={channels} batch={batch}   [{prov}]", flush=True)
        print(f"  {'engine':<24}{'wall us':>10}{'HBM MB':>10}", flush=True)

        engines = [("reim+partial", "f32", lambda w_, z: spectral_conv(w_, z, m)),
                   ("reim+partial", "bf16", lambda w_, z: spectral_conv(w_, z, m, BF)),
                   ("reim full+select", "f32", lambda w_, z: spectral_2corner_hybrid(w_, z, m)),
                   ("reim hermitian", "f32", lambda w_, z: spectral_2corner_herm(w_, z, m)),
                   ("jnp.fft", "f32", lambda w_, z: reference.spectral_conv(w_, z, m))]
        for name, dt, fn in engines:
            tag = f"{label}_{name}_{dt}".replace(" ", "").replace("^", "")
            try:
                wall, hbm, trace = run_one(tag, fn, w, x)
            except Exception as exc:
                print(f"  {name+' '+dt:<24}  FAILED {type(exc).__name__}: {str(exc)[:60]}", flush=True)
                continue
            print(f"  {name+' '+dt:<24}{wall:>10.1f}{hbm/1e6:>10.1f}", flush=True)
            rows.append({"case": label, "grid": grid, "n_modes": m, "channels": channels,
                         "batch": batch, "engine": name, "dtype": dt, "wall_us": wall,
                         "hbm_mb": hbm / 1e6, "per_sample_us": wall / batch,
                         "trace": os.path.relpath(trace, LOGDIR), "provenance": prov})

    with open("/tmp/docs_fno_real.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        wr.writeheader(); wr.writerows(rows)
    print(f"\nwrote /tmp/docs_fno_real.csv ({len(rows)} rows)", flush=True)
    print("FNO_REAL_DONE", flush=True)


if __name__ == "__main__":
    main()
