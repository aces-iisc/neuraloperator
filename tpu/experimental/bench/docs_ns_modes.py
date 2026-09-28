"""num_modes sweep for Navier-Stokes, the one dataset the earlier sweep skipped."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fno import data, train
from ...fno.layers import n_params

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
CSV = os.environ.get("DOCS_NSM_CSV", "/tmp/docs_ns_modes.csv")
OUT = os.environ.get("DOCS_NSM_OUT", "/tmp/docs_ns_modes")
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "60"))
MODES = [int(v) for v in os.environ.get("DOCS_MODES", "8,16,32,64").split(",")]
FIELDS = ["case", "n_modes", "channels", "batch", "epochs", "rel_l2", "train_s",
          "infer_us", "params", "status"]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"NS-128 num_modes sweep {MODES}, {EPOCHS} epochs, channels=64, batch=8", flush=True)
    os.makedirs(OUT, exist_ok=True)
    xtr, ytr, xte, yte = data.load_ns(ROOT, 128, n_train=10000, n_test=1000)
    print(f"train{tuple(xtr.shape)} test{tuple(xte.shape)}\n", flush=True)
    print(f"  {'n_modes':>8}{'rel L2':>13}{'train s':>10}{'us/sample':>11}{'params':>14}", flush=True)
    rows = []
    for m in MODES:
        try:
            P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=m, channels=64, n_layers=4,
                                       steps=EPOCHS, batch=8, verbose=False,
                                       save=os.path.join(OUT, f"NS128_m{m}.npz"))
            us = train.infer_us_per_sample(P, xte, m)
            print(f"  {m:>8}{rel:>13.4e}{secs:>10.1f}{us:>11.2f}{n_params(P):>14,}", flush=True)
            rows.append({"case": "NS-128", "n_modes": m, "channels": 64, "batch": 8,
                         "epochs": EPOCHS, "rel_l2": rel, "train_s": secs, "infer_us": us,
                         "params": n_params(P), "status": "ok"})
        except Exception as exc:
            print(f"  {m:>8}  {type(exc).__name__}: {str(exc)[:50]}", flush=True)
            rows.append({"case": "NS-128", "n_modes": m, "channels": 64, "batch": 8,
                         "epochs": EPOCHS, "rel_l2": float("nan"), "train_s": float("nan"),
                         "infer_us": float("nan"), "params": 0, "status": type(exc).__name__})
        with open(CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print("NSMODES_DONE", flush=True)


if __name__ == "__main__":
    main()
