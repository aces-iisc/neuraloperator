"""Coupled and reduced packings of the row transform, against the plain partial one."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno.spectral import _E, _dft_rows

CSV = os.environ.get("DOCS_PACK_CSV", "/tmp/docs_pack.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_pack")
REPS = 20
CHANNELS = 24
BATCH = 128
GRIDS = (16, 32, 64, 128)
N_MODES = 16
FIELDS = ["n_grid", "n_modes", "variant", "us_per_sample", "rel_err", "status", "trace"]


# The row passes live in docs_matrix so there is one implementation of each, and this bench
# times exactly the code the sweep and the layer use.  The older copies here built the full
# spectrum and sliced, which is not what any of them do any more.
from .docs_matrix import ROWS, band_of as row_band_of          # noqa: E402,F401

VARIANTS = dict(ROWS)


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = highest, C={CHANNELS}, B={BATCH}, n_modes={N_MODES}\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for n_grid in GRIDS:
        x = jax.random.normal(jax.random.PRNGKey(0), (BATCH, CHANNELS, n_grid, n_grid), F)
        print(f"--- grid {n_grid}x{n_grid}", flush=True)
        ref = None
        for name, fn in VARIANTS.items():
            tag = f"g{n_grid}_{name}"
            try:
                f = jax.jit(lambda z, _f=fn: _f(z, N_MODES, F))
                got = jax.block_until_ready(f(x))
                if ref is None:
                    ref = [np.asarray(g) for g in got]
                    err = 0.0
                else:
                    den = sum(float(np.sqrt(np.sum(r ** 2))) for r in ref)
                    err = sum(float(np.sqrt(np.sum((np.asarray(g) - r) ** 2)))
                              for g, r in zip(got, ref)) / den
                ts = []
                with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        jax.block_until_ready(f(x))
                        ts.append(time.perf_counter() - t0)
                us = sorted(ts)[REPS // 2] * 1e6 / BATCH
                print(f"    {name:<9}{us:>9.3f} us/sample   err {err:.2e}", flush=True)
                rows.append({"n_grid": n_grid, "n_modes": N_MODES, "variant": name,
                             "us_per_sample": us, "rel_err": err, "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"    {name:<9}{why:>9}  {str(exc)[:90]}", flush=True)
                rows.append({"n_grid": n_grid, "n_modes": N_MODES, "variant": name,
                             "us_per_sample": float("nan"), "rel_err": float("nan"),
                             "status": why, "trace": ""})
            save(rows)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/pack.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("\nPACK_DONE", flush=True)


if __name__ == "__main__":
    main()
