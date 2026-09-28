"""Times the spectral-conv matmul variants (block / Karatsuba mix, stacked cos+sin) on device."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno import layers, spectral
from ..fno.gemm_variants import spectral_conv_variant

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_gemm")
CSV = os.environ.get("DOCS_GEMM_CSV", "/tmp/docs_gemm.csv")
FIELDS = ["case", "variant", "wall_us", "rel_diff", "status", "trace"]
REPS = 20

# (label, batch, channels, grid, n_modes) -- the shapes the trained models actually use
CASES = [
    ("Darcy-32", 8, 24, 32, 16),
    ("Darcy-64", 8, 24, 64, 16),
    ("NS-128", 8, 64, 128, 64),
]
VARIANTS = [("current", None, None),
            ("naive+stacked", "naive", True),
            ("block", "block", False),
            ("block+stacked", "block", True),
            ("karatsuba", "karatsuba", False),
            ("karatsuba+stacked", "karatsuba", True)]


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
    for label, B, C, g, nm in CASES:
        w = layers._init_spectral_weight(jax.random.PRNGKey(0), C, C, nm)
        x = jnp.asarray(np.random.default_rng(0).standard_normal((B, C, g, g)).astype(np.float32))
        ref = np.asarray(spectral.spectral_conv(w, x, nm), np.float64)
        den = np.max(np.abs(ref))
        print(f"\n### {label}  B={B} channels={C} grid={g}x{g} n_modes={nm}", flush=True)
        print(f"  {'variant':<20}{'wall us':>10}{'rel diff':>11}", flush=True)
        for name, mix, st in VARIANTS:
            tag = f"{label}_{name}".replace("+", "_")
            try:
                fn = ((lambda z: spectral.spectral_conv(w, z, nm)) if mix is None else
                      (lambda z, m=mix, s=st: spectral_conv_variant(w, z, nm, mix=m, stacked=s)))
                wall = run(tag, fn, x)
                d = float(np.max(np.abs(np.asarray(jax.jit(fn)(x), np.float64) - ref))/den)
                print(f"  {name:<20}{wall:>10.1f}{d:>11.1e}", flush=True)
                rows.append({"case": label, "variant": name, "wall_us": wall, "rel_diff": d,
                             "status": "ok", "trace": tag})
            except Exception as exc:
                why = type(exc).__name__
                print(f"  {name:<20}{why:>10}   {str(exc)[:44]}", flush=True)
                rows.append({"case": label, "variant": name, "wall_us": float("nan"),
                             "rel_diff": float("nan"), "status": why, "trace": ""})
            save(rows)
    save(rows)
    print("GEMM_DONE", flush=True)


if __name__ == "__main__":
    main()
