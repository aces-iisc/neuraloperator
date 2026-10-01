"""Inference at the training batch and at B* = 128, on real data with the trained weights."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fno import data, model

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_bstar")
CSV = os.environ.get("DOCS_BSTAR_CSV", "/tmp/docs_bstar.csv")
WEIGHTS = os.environ.get("DOCS_WEIGHTS",
                         os.path.expanduser(os.environ.get("REPRO_WEIGHTS",
                                                           "~/repro_out/weights")))
ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
FIELDS = ["case", "batch", "kind", "dev_ms_total", "us_per_sample", "rel_l2", "n_eval",
          "two_c", "b_star_predicted", "status", "trace"]
REPS = 10
B_STAR = int(os.environ.get("DOCS_BSTAR", "128"))
BATCHES = [int(v) for v in os.environ.get("DOCS_BATCHES", "").split(",") if v.strip()]

# (label, loader, channels, n_modes, training batch)
CASES = [
    ("Burgers",  lambda: data.load_burgers(ROOT),          24, 16, 16),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32),        24, 16, 8),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64),        24, 16, 8),
    ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=16, n_test=512), 64, 64, 8),
]


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def timed(fn, x, tag):
    j = jax.jit(fn)
    jax.block_until_ready(j(x))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(x)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    print(f"B* rule: 2C < 128 -> B* = {B_STAR};  2C = 128 -> keep the small batch\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    print(f"  {'case':<10}{'batch':>6}{'kind':>10}{'us/sample':>11}{'rel L2':>11}{'2C':>5}", flush=True)
    for label, loader, C, nm, B_train in CASES:
        try:
            _, _, xte, yte = loader()
        except Exception as exc:
            print(f"  {label:<10} data SKIPPED: {type(exc).__name__}", flush=True); continue
        wpath = os.path.join(WEIGHTS, f"{label}_ours.npz")
        if not os.path.exists(wpath):
            print(f"  {label:<10} weights missing at {wpath}", flush=True); continue
        P = model.load_params(wpath, channels=C, n_modes=nm, n_layers=4)
        two_c = 2 * C
        predicted = B_STAR if two_c < 128 else B_train
        for B in (BATCHES or sorted({B_train, B_STAR})):
            n_eval = (len(xte) // B) * B
            if n_eval == 0:
                print(f"  {label:<10}{B:>6}  too few test samples", flush=True); continue
            kind = "training" if B == B_train else ("B*" if B == predicted else f"B={B}")
            tag = f"{label}_{B}"
            try:
                xb = jnp.asarray(np.asarray(xte[:B], np.float32))
                t = timed(lambda z: model.forward(P, z, nm), xb, tag)
                # accuracy over the whole test set at this batch
                errs, dens = 0.0, 0.0
                for s in range(0, n_eval, B):
                    p = np.asarray(model.forward(P, jnp.asarray(np.asarray(xte[s:s+B], np.float32)), nm))
                    g = np.asarray(yte[s:s+B], np.float32)
                    errs += float(np.sum((p - g) ** 2)); dens += float(np.sum(g ** 2))
                rel = (errs / dens) ** 0.5
                us = t * 1e6 / B
                print(f"  {label:<10}{B:>6}{kind:>10}{us:>11.2f}{rel:>11.4e}{two_c:>5}", flush=True)
                rows.append({"case": label, "batch": B, "kind": kind, "dev_ms_total": t*1e3,
                             "us_per_sample": us, "rel_l2": rel, "n_eval": n_eval,
                             "two_c": two_c, "b_star_predicted": predicted, "status": "ok",
                             "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"  {label:<10}{B:>6}{why:>10}   {str(exc)[:40]}", flush=True)
                rows.append({"case": label, "batch": B, "kind": "fail", "dev_ms_total": float("nan"),
                             "us_per_sample": float("nan"), "rel_l2": float("nan"), "n_eval": 0,
                             "two_c": two_c, "b_star_predicted": predicted, "status": why,
                             "trace": ""})
            save(rows)
        print(flush=True)
    save(rows)
    print("BSTAR_DONE", flush=True)


if __name__ == "__main__":
    main()
