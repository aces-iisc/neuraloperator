"""Baseline config against our autotuned one, plus a bf16 batch sweep."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import layers, data, model, spectral
from ..fno.reference import forward as jnp_forward

CSV = os.environ.get("DOCS_FINAL_CSV", "/tmp/docs_final.csv")
ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
WEIGHTS = os.path.expanduser(os.environ.get("REPRO_WEIGHTS", "~/repro_out/weights"))
FIELDS = ["part", "case", "config", "dtype", "batch", "us_per_sample",
          "loss", "error", "h1", "l2", "status"]
REPS = 20
PAPER_B = 128                      # train.INFER_CAP, what the baseline numbers use

LOSS = {"Burgers": "l2", "Darcy-32": "h1", "Darcy-64": "h1", "NS-128": "h1"}
CASES = [("Burgers",  lambda: data.load_burgers(ROOT),   24, 16),
         ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 24, 16),
         ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 24, 16),
         ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=16, n_test=512), 64, 64)]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def rel_errors(f, xte, yte, B, n_eval):
    """Relative H1 and L2 over the test split, chunked.

    Both are reported: the model is judged on the objective its config trains with, and the
    other is carried because the FNO literature quotes relative L2 and a table with only one
    of them cannot be placed against it.
    """
    totals = {"h1": [0.0, 0.0], "l2": [0.0, 0.0]}
    for start in range(0, n_eval, B):
        pred = f(jnp.asarray(np.asarray(xte[start:start + B], np.float32)))
        gold = jnp.asarray(np.asarray(yte[start:start + B], np.float32))
        for name, fn in (("h1", layers.h1_rel), ("l2", layers.lp_rel)):
            totals[name][0] += float(fn(pred, gold))
            totals[name][1] += gold.shape[0]
    # both losses sum over the batch, so divide by the count to get a per-sample figure
    return {k: v[0] / max(1, v[1]) for k, v in totals.items()}


def wall(f, x):
    jax.block_until_ready(f(x))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); jax.block_until_ready(f(x)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]*1e6/x.shape[0]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    rows = []
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print("\n=== A. bf16 batch sweep: does halving the bytes allow a bigger batch? ===", flush=True)
    print(f"  {'case':<10}{'dtype':>7}{'B':>6}{'wall us/sample':>16}", flush=True)
    bstar = {}
    for label, loader, C, nm in CASES:
        try: _, _, xte, yte = loader()
        except Exception as e: print(f"  {label}: skipped"); continue
        P = model.load_params(os.path.join(WEIGHTS, f"{label}_ours.npz"), channels=C,
                              n_modes=nm, n_layers=4)
        for dt, dn in ((F, "f32"), (BF, "bf16")):
            best = (None, 1e18)
            for B in batches:
                if B > len(xte): continue
                try:
                    x = jnp.asarray(np.asarray(xte[:B], np.float32))
                    us = wall(jax.jit(lambda z: model.forward(P, z, nm, dt)), x)
                    if us < best[1]: best = (B, us)
                    rows.append({"part": "A", "case": label, "config": "sweep", "dtype": dn,
                                 "batch": B, "us_per_sample": us, "loss": LOSS.get(label, "h1"), "error": float("nan"),
                             "h1": float("nan"), "l2": float("nan"),
                                 "status": "ok"})
                except Exception as exc:
                    rows.append({"part": "A", "case": label, "config": "sweep", "dtype": dn,
                                 "batch": B, "us_per_sample": float("nan"),
                                 "loss": LOSS.get(label, "h1"), "error": float("nan"),
                             "h1": float("nan"), "l2": float("nan"), "status": type(exc).__name__})
                save(rows)
            bstar[(label, dn)] = best[0]
            print(f"  {label:<10}{dn:>7}{best[0]:>6}{best[1]:>16.2f}   <- best", flush=True)
    print("\n  B* per dtype:", {k: v for k, v in bstar.items()}, flush=True)

    print("\n=== B. baseline config against ours ===", flush=True)
    print(f"  {'case':<10}{'config':<26}{'B':>5}{'wall us/sample':>16}{'rel L2':>12}", flush=True)
    for label, loader, C, nm in CASES:
        try: _, _, xte, yte = loader()
        except Exception: continue
        P = model.load_params(os.path.join(WEIGHTS, f"{label}_ours.npz"), channels=C,
                              n_modes=nm, n_layers=4)
        Pb = model.load_params(os.path.join(WEIGHTS, f"{label}_jnp.npz"), channels=C,
                               n_modes=nm, n_layers=4)
        setups = [("baseline (jnp.fft, B=128)", Pb, jnp_forward, F, PAPER_B),
                  ("baseline (jnp.fft, B*)", Pb, jnp_forward, F, bstar.get((label, "f32"), PAPER_B)),
                  ("ours f32 (B*)", P, model.forward, F, bstar.get((label, "f32"), PAPER_B)),
                  ("ours bf16 (B*)", P, model.forward, BF, bstar.get((label, "bf16"), PAPER_B))]
        for cname, params, fwd, dt, B in setups:
            n_eval = (len(xte)//B)*B
            if n_eval == 0: continue
            try:
                f = jax.jit(lambda z, _p=params, _f=fwd, _d=dt:
                            _f(_p, z, nm) if _f is jnp_forward else _f(_p, z, nm, _d))
                x = jnp.asarray(np.asarray(xte[:B], np.float32))
                us = wall(f, x); errs = rel_errors(f, xte, yte, B, n_eval)
                loss = LOSS.get(label, "h1")
                print(f"  {label:<10}{cname:<26}{B:>5}{us:>16.2f}"
                      f"{errs[loss]:>12.4e} ({loss})", flush=True)
                rows.append({"part": "B", "case": label, "config": cname, "dtype": str(dt.__name__),
                             "batch": B, "us_per_sample": us, "loss": loss,
                             "error": errs[loss], "h1": errs["h1"], "l2": errs["l2"],
                             "status": "ok"})
            except Exception as exc:
                print(f"  {label:<10}{cname:<26}{B:>5}{type(exc).__name__:>16}", flush=True)
                rows.append({"part": "B", "case": label, "config": cname, "dtype": str(dt.__name__),
                             "batch": B, "us_per_sample": float("nan"), "loss": LOSS.get(label, "h1"), "error": float("nan"),
                             "h1": float("nan"), "l2": float("nan"),
                             "status": type(exc).__name__})
            save(rows)
        print(flush=True)
    save(rows); print("FINAL_DONE", flush=True)


if __name__ == "__main__":
    main()
