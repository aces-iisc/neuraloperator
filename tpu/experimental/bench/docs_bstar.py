"""Inference at the training batch and at B*, on real data with the trained weights.

Three configurations per dataset, so that the speedup separates what the rewrites bought from
what the batch bought: the baseline transform at the training batch, our layer at the same batch,
and our layer at B*.  Errors are reported in both norms, each as a per-sample mean, because the
loss the model was trained on differs by dataset.
"""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fno import data, layers, model
from ..fno import reference

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_bstar")
CSV = os.environ.get("DOCS_BSTAR_CSV", "/tmp/docs_bstar.csv")
WEIGHTS = os.environ.get("DOCS_WEIGHTS",
                         os.path.expanduser(os.environ.get("REPRO_WEIGHTS",
                                                           "~/repro_out/weights")))
ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
FIELDS = ["case", "engine", "batch", "kind", "loss", "dev_ms_total", "us_per_sample",
          "error", "h1", "l2", "n_eval", "two_c", "b_star_predicted", "status", "trace"]
REPS = 10
B_STAR = int(os.environ.get("DOCS_BSTAR", "128"))
BATCHES = [int(v) for v in os.environ.get("DOCS_BATCHES", "").split(",") if v.strip()]

# (label, loader, channels, n_modes, training batch, training objective)
CASES = [
    ("Burgers",  lambda: data.load_burgers(ROOT),          24, 16, 16, "l2"),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32),        24, 16, 8, "h1"),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64),        24, 16, 8, "h1"),
    ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=16, n_test=512), 64, 64, 8, "h1"),
]


def errors(forward, xte, yte, batch, n_eval):
    """Per-sample mean relative error in both norms over the first `n_eval` test samples.

    The loss functions reduce with a sum over the batch, matching the torch objects they
    reproduce, so the running total is divided by the sample count."""
    totals = {"h1": 0.0, "l2": 0.0}
    for start in range(0, n_eval, batch):
        pred = forward(jnp.asarray(np.asarray(xte[start:start + batch], np.float32)))
        gold = jnp.asarray(np.asarray(yte[start:start + batch], np.float32))
        totals["h1"] += float(layers.h1_rel(pred, gold))
        totals["l2"] += float(layers.lp_rel(pred, gold))
    return {k: v / n_eval for k, v in totals.items()}


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
    print(f"  {'case':<10}{'kind':<11}{'engine':<9}{'batch':>6}{'us/sample':>11}"
          f"{'error':>11}{'2C':>8}", flush=True)
    for label, loader, C, nm, B_train, loss in CASES:
        try:
            _, _, xte, yte = loader()
        except Exception as exc:
            print(f"  {label:<10} data SKIPPED: {type(exc).__name__}", flush=True); continue
        wpath = os.path.join(WEIGHTS, f"{label}_ours.npz")
        if not os.path.exists(wpath):
            print(f"  {label:<10} weights missing at {wpath}", flush=True); continue
        P = model.load_params(wpath, channels=C, n_modes=nm, n_layers=4)
        bpath = os.path.join(WEIGHTS, f"{label}_jnp.npz")
        Pb = (model.load_params(bpath, channels=C, n_modes=nm, n_layers=4)
              if os.path.exists(bpath) else P)
        two_c = 2 * C
        predicted = B_STAR if two_c < 128 else B_train

        # baseline at the training batch, then ours at the same batch, then ours at B*
        setups = [("baseline", "jnp.fft", Pb, reference.forward, B_train),
                  ("+rewrites", "ours", P, model.forward, B_train)]
        if predicted != B_train:
            setups.append(("+B*", "ours", P, model.forward, predicted))
        for kind, engine, params, fwd, B in (
                [(f"B={b}", "ours", P, model.forward, b) for b in BATCHES] or setups):
            n_eval = (len(xte) // B) * B
            if n_eval == 0:
                print(f"  {label:<10}{B:>6}  too few test samples", flush=True); continue
            tag = f"{label}_{engine}_{B}_{kind}".replace("*", "star").replace("+", "")
            try:
                f = jax.jit(lambda z, _p=params, _f=fwd: _f(_p, z, nm))
                xb = jnp.asarray(np.asarray(xte[:B], np.float32))
                t = timed(lambda z, _f=f: _f(z), xb, tag)
                errs = errors(f, xte, yte, B, n_eval)
                us = t * 1e6 / B
                print(f"  {label:<10}{kind:<11}{engine:<9}{B:>6}{us:>11.2f}"
                      f"{errs[loss]:>11.4e} ({loss}){two_c:>5}", flush=True)
                rows.append({"case": label, "engine": engine, "batch": B, "kind": kind,
                             "loss": loss, "dev_ms_total": t*1e3, "us_per_sample": us,
                             "error": errs[loss], "h1": errs["h1"], "l2": errs["l2"],
                             "n_eval": n_eval, "two_c": two_c, "b_star_predicted": predicted,
                             "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"  {label:<10}{kind:<11}{engine:<9}{B:>6}{why:>11}   {str(exc)[:40]}", flush=True)
                rows.append({"case": label, "engine": engine, "batch": B, "kind": "fail",
                             "loss": loss, "dev_ms_total": float("nan"),
                             "us_per_sample": float("nan"), "error": float("nan"),
                             "h1": float("nan"), "l2": float("nan"), "n_eval": 0,
                             "two_c": two_c, "b_star_predicted": predicted, "status": why,
                             "trace": ""})
            save(rows)
        print(flush=True)
    save(rows)
    print("BSTAR_DONE", flush=True)


if __name__ == "__main__":
    main()
