"""Inference on the trained weights, measured the way the paper measures it: wall clock at
batch 128, the same relative-L2 definition as train.test_rel_l2, f32 and bf16."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import data, model, spectral
from ..fno.gemm_variants import spectral_conv_variant

CSV = os.environ.get("DOCS_INFER_CSV", "/tmp/docs_infer.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_infer")
WEIGHTS = os.path.expanduser("~/nop/tpu/results/baseline")
ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
FIELDS = ["case", "layer", "dtype", "batch", "us_per_sample", "rel_l2", "n_eval", "status", "trace"]
REPS = 10

# (label, loader, channels, n_modes, train batch, B*)
# (label, loader, channels, n_modes, paper batch = INFER_CAP, B*)
CASES = [
    ("Burgers",  lambda: data.load_burgers(ROOT),   24, 16, 128, 128),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 24, 16, 128, 128),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 24, 16, 128, 128),
    ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=16, n_test=512), 64, 64, 128, 8),
]
LAYERS = [("original", dict(mix="naive", stacked=False), F),
          ("block+stacked", None, F),
          ("block+stacked bf16", None, BF)]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    orig = spectral.spectral_conv
    print(f"  {'case':<10}{'layer':<20}{'B':>5}{'wall us/sample':>16}{'rel L2':>12}", flush=True)
    for label, loader, C, nm, Btr, Bst in CASES:
        try:
            _, _, xte, yte = loader()
        except Exception as exc:
            print(f"  {label}: data skipped ({type(exc).__name__})", flush=True); continue
        wp = os.path.join(WEIGHTS, f"{label}_ours.npz")
        if not os.path.exists(wp):
            print(f"  {label}: weights missing", flush=True); continue
        P = model.load_params(wp, channels=C, n_modes=nm, n_layers=4)
        for lname, kw, dt in LAYERS:
            spectral.spectral_conv = orig if kw is None else (
                lambda w, x, n, dtype=F, _k=kw: spectral_conv_variant(w, x, n, dtype=dtype, **_k))
            model.spectral_conv = spectral.spectral_conv
            for B in sorted({Btr, Bst}):
                n_eval = (len(xte) // B) * B
                if n_eval == 0: continue
                tag = f"{label}_{lname.replace('+','_').replace(' ','_')}_{B}"
                try:
                    xb = jnp.asarray(np.asarray(xte[:B], np.float32))
                    f = jax.jit(lambda z: model.forward(P, z, nm, dt))
                    jax.block_until_ready(f(xb))
                    ts = []
                    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                        for _ in range(REPS):
                            t0 = time.perf_counter(); jax.block_until_ready(f(xb))
                            ts.append(time.perf_counter()-t0)
                    e, dn = 0.0, 0.0        # same definition as train.test_rel_l2
                    for s in range(0, n_eval, B):
                        p = np.asarray(f(jnp.asarray(np.asarray(xte[s:s+B], np.float32))))
                        g = np.asarray(yte[s:s+B], np.float32)
                        ax = tuple(range(1, g.ndim))
                        e += float(np.sum(np.sqrt(np.sum((p-g)**2, axis=ax))))
                        dn += float(np.sum(np.sqrt(np.sum(g**2, axis=ax))))
                    rel = e/dn
                    us = sorted(ts)[len(ts)//2]*1e6/B
                    print(f"  {label:<10}{lname:<20}{B:>5}{us:>16.2f}{rel:>12.4e}", flush=True)
                    rows.append({"case": label, "layer": lname, "dtype": str(dt.__name__),
                                 "batch": B, "us_per_sample": us, "rel_l2": rel,
                                 "n_eval": n_eval, "status": "ok", "trace": tag})
                except Exception as exc:
                    why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    print(f"  {label:<10}{lname:<20}{B:>5}{why:>16}", flush=True)
                    rows.append({"case": label, "layer": lname, "dtype": str(dt.__name__),
                                 "batch": B, "us_per_sample": float("nan"),
                                 "rel_l2": float("nan"), "n_eval": 0, "status": why, "trace": ""})
                save(rows)
        print(flush=True)
    spectral.spectral_conv = orig
    save(rows); print("INFER_DONE", flush=True)


if __name__ == "__main__":
    main()
