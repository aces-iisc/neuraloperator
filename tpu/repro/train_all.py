"""Train one model per dataset, once, and keep the weights.

Every table that needs a trained network reads these files rather than training its own, so a
full reproduction trains four times, not once per table.  Training is on H1, the FNO objective.
"""
import os, time

import jax
import numpy as np

from ..fno import model, train
from . import common

TABLE = None            # produces weights, not a table
EPOCHS = int(os.environ.get("REPRO_EPOCHS", "60"))


def run():
    print(f"\n### trained weights ({EPOCHS} epochs, H1 objective)", flush=True)
    common.ensure_dirs()
    made = []
    for label, _, channels, n_modes, _ in common.CASES:
        path = common.weight_file(label)
        if os.path.exists(path):
            print(f"  {label}: already trained, reusing {path}", flush=True)
            made.append(path)
            continue
        loaded = common.load_case(label)
        if loaded is None:
            continue
        x_tr, y_tr, x_te, y_te = loaded
        t0 = time.perf_counter()
        P, err, _ = train.train(x_tr, y_tr, x_te, y_te, n_modes=n_modes, channels=channels,
                                n_layers=4, steps=EPOCHS, batch=32, loss="h1", verbose=False)
        model.save_params(P, path)
        print(f"  {label}: H1 {err:.4e}  ({time.perf_counter() - t0:.0f} s) -> {path}",
              flush=True)
        made.append(path)
    return made


if __name__ == "__main__":
    run()
