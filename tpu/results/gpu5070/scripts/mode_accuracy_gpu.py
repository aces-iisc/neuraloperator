"""What the kept band buys in accuracy: train at each band and measure the test error.

The companion to the cost mode sweep, and the half of the question that sweep cannot answer. A
wider band always costs more and the cost sweep says how much; whether it is worth paying is a
property of the problem, so it has to be trained for. Every band here gets its own training run
under one protocol -- same epochs, same optimiser, same seed, same split -- so the only thing
that differs between rows is the band.

The error reported is the best test error over the run rather than the last, matching how the
shipped weights were selected, and both metrics are kept: the H1 norm is what the model trains
against and the relative L2 is what the FNO literature quotes.

Thirty epochs is not convergence. It is enough to rank the bands against each other, which is
what the figure is for, and it is the budget the TPU mode study used, so the two are comparable.
Absolute errors here are therefore not the numbers to quote as the model's accuracy.
"""
import csv
import os
import sys
import time

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.fno.data import load_burgers, load_darcy, load_ns
from torch_fno.fno.train import evaluate, train

ROOT = os.environ.get("FNO_DATA", "/home/anirudh/fno/data")
OUT = os.environ.get("MODE_ACC_OUT", "/home/anirudh/fno/mode_accuracy_gpu.csv")
EPOCHS = int(os.environ.get("MODE_ACC_EPOCHS", 30))
FIELDS = ["case", "grid", "channels", "n_modes", "epochs", "batch", "params",
          "rel_h1", "rel_l2", "train_s"]

# The bands are a subset of the ones the cost sweep measured, so the two join on (case, n_modes)
# without interpolating either.  Each run is a full training, so the grids stop where the cost of
# another point stops being worth the resolution it adds -- soonest on Navier--Stokes, where one
# run is a third of an hour.
CASES = [
    ("Burgers", lambda: load_burgers(ROOT), 17, 24, 16, [2, 4, 6, 8, 10, 12, 14, 16]),
    ("Darcy-32", lambda: load_darcy(ROOT, 32), 32, 24, 16, [4, 8, 12, 16, 20, 24, 28, 32]),
    ("Darcy-64", lambda: load_darcy(ROOT, 64), 64, 24, 16,
     [4, 8, 12, 16, 20, 24, 32, 40, 48, 64]),
    ("NS-128", lambda: load_ns(ROOT, 128), 128, 64, 16, [8, 16, 24, 32, 48, 64]),
]


def done_already(path):
    """Rows already on disk, so a long run can be resumed rather than restarted."""
    if not os.path.exists(path):
        return set()
    return {(r["case"], int(r["n_modes"])) for r in csv.DictReader(open(path))}


def append(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def main():
    seen = done_already(OUT)
    for name, loader, grid, channels, batch, bands in CASES:
        todo = [m for m in bands if (name, m) not in seen]
        if not todo:
            print(f"{name}: all bands already measured", flush=True)
            continue
        xtr, ytr, xte, yte = loader()
        for modes in todo:
            start = time.perf_counter()
            module, best_h1, _ = train(xtr, ytr, xte, yte, n_modes=modes, channels=channels,
                                       epochs=EPOCHS, batch=batch, loss="h1", seed=0,
                                       device="cuda", verbose=False,
                                       name=f"{name}_m{modes}")
            # The L2 is read off the same selected weights, so the two columns describe one model
            # rather than two runs that happened to stop at different points.
            best_l2 = evaluate(module, xte, yte, "l2", device="cuda")
            elapsed = time.perf_counter() - start
            append(OUT, {"case": name, "grid": grid, "channels": channels, "n_modes": modes,
                         "epochs": EPOCHS, "batch": batch, "params": module.n_params(),
                         "rel_h1": f"{best_h1:.6f}", "rel_l2": f"{best_l2:.6f}",
                         "train_s": f"{elapsed:.1f}"})
            print(f"{name:9} m={modes:3d}  params={module.n_params():>10,}  "
                  f"H1={best_h1:.4f}  L2={best_l2:.4f}  {elapsed:6.1f}s", flush=True)
            del module
            torch.cuda.empty_cache()
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
