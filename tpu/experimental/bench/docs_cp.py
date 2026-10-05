"""How much of TFNO's parameter saving the CP contraction actually delivers.

The rank-R filter stores 128x fewer numbers than the dense one on Darcy and still does not run
faster.  That is a property of how the contraction is issued, not of the decomposition: the
published form applies the operator four times (Wre*fre, Wim*fim, Wim*fre, Wre*fim), which is
eight einsums whose contracted axis is the rank itself -- 8 wide on a 128-wide unit.

This times four ways of issuing the same arithmetic, checks they agree, and writes a trace for
each so the device time can be split across the matrix unit and the copies around it.
"""
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import F
from ..fno import factorized as fz
from ..fno import factorized_opt as fzo

CSV = os.environ.get("DOCS_CP_CSV", "/tmp/docs_cp.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_cp")
REPS = 20
RANK = int(os.environ.get("DOCS_CP_RANK", "8"))
FIELDS = ["case", "channels", "n_modes", "batch", "rank", "variant", "family",
          "einsums", "params", "us_per_call", "rel_err", "status", "trace"]

# (label, channels, n_modes, batch) -- the shapes the paper reports
CASES = [("Darcy-32", 24, 16, 32), ("Darcy-64", 24, 16, 32), ("NS-128", 64, 64, 4)]

# (name, filter family, einsums issued, init, prepare, apply).  The dense and F-FNO rows are the
# comparison TFNO has to beat: all three carry the Section 5 optimisations, so a difference
# between them is a difference between filters rather than between implementations.
VARIANTS = [
    ("dense",      "dense", 1, fz.dense_init, fzo.dense_block,         fzo.dense_filter_opt),
    ("f-fno",      "ffno",  2, fz.sep_init,   fzo.sep_block,           fzo.sep_filter_opt),
    ("published",  "tfno",  8, fz.cp_init,    lambda w: w,             fz.cp_filter),
    ("adio",       "tfno",  8, fz.cp_init,    fzo.cp_block,            fzo.cp_filter_opt),
    ("stacked",    "tfno",  5, fz.cp_init,    fzo.cp_block_stacked,    fzo.cp_filter_stacked),
    ("fused",      "tfno",  2, fz.cp_init,    fzo.cp_block_fused,      fzo.cp_filter_fused),
    ("fused2",     "tfno",  2, fz.cp_init,    fzo.cp_block_fused2,     fzo.cp_filter_fused2),
]


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def timed(fn, args, tag):
    jax.block_until_ready(fn(*args))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
    return sorted(ts)[REPS // 2] * 1e6


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = {jax.config.jax_default_matmul_precision} | rank = {RANK}\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []

    for label, channels, n_modes, batch in CASES:
        key = jax.random.PRNGKey(0)
        half = n_modes // 2
        shape = (batch, channels, n_modes, half)

        def weights_for(family, init):
            """Each family stores a different thing; trim the mode axes the layer truncates."""
            if family == "tfno":
                w = init(key, channels, channels, n_modes, rank=RANK)
                for part in ("re_", "im_"):
                    w[part + "D"] = w[part + "D"][:half]
            elif family == "ffno":
                w = init(key, channels, channels, n_modes)
                for part in ("lr", "li"):
                    w[part] = w[part][:, :, :half] if w[part].ndim == 3 else w[part][:half]
            else:
                w = init(key, channels, channels, n_modes)
                for k in ("w_re", "w_im"):
                    w[k] = w[k][..., :half]
            return w
        fr = jax.random.normal(jax.random.PRNGKey(1), shape, F)
        fi = jax.random.normal(jax.random.PRNGKey(2), shape, F)

        print(f"--- {label}: C={channels} m={n_modes} B={batch} "
              f"field {shape}", flush=True)
        print(f"    {'variant':<12}{'family':<7}{'einsums':>8}{'params':>11}{'us/call':>10}",
              flush=True)
        refs = {}

        for name, family, n_einsum, init, prepare, apply_fn in VARIANTS:
            tag = f"{label}_{name}"
            try:
                weight = weights_for(family, init)
                n_par = int(sum(np.asarray(v).size for v in weight.values()))
                prepared = prepare(weight)
                fn = jax.jit(lambda a, b, _w=prepared, _f=apply_fn: _f(_w, a, b))
                got = jax.block_until_ready(fn(fr, fi))
                # correctness is only meaningful within a family: the three filters are
                # different models and are not expected to agree with one another
                if family not in refs:
                    refs[family] = [np.asarray(g, np.float64) for g in got]
                    err = 0.0
                else:
                    num = sum(float(np.linalg.norm(np.asarray(g, np.float64) - r))
                              for g, r in zip(got, refs[family]))
                    den = sum(float(np.linalg.norm(r)) for r in refs[family])
                    err = num / den
                us = timed(fn, (fr, fi), tag)
                print(f"    {name:<12}{family:<7}{n_einsum:>8}{n_par:>11,}{us:>10.2f}",
                      flush=True)
                rows.append({"case": label, "channels": channels, "n_modes": n_modes,
                             "batch": batch, "rank": RANK, "variant": name, "family": family,
                             "einsums": n_einsum, "params": n_par, "us_per_call": us,
                             "rel_err": err, "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"    {name:<12}{family:<7}{n_einsum:>8}{why:>21}  {str(exc)[:50]}",
                      flush=True)
                rows.append({"case": label, "channels": channels, "n_modes": n_modes,
                             "batch": batch, "rank": RANK, "variant": name, "family": family,
                             "einsums": n_einsum, "params": 0, "us_per_call": float("nan"),
                             "rel_err": float("nan"), "status": why, "trace": ""})
            save(rows)
        print(flush=True)

    save(rows)
    print("CP_DONE", flush=True)


if __name__ == "__main__":
    main()
