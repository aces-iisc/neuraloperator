"""How much does the same band move between training runs? The threshold the knee is taken at.

The accuracy-against-band curve falls steeply and then flattens, and on the flat the differences
between neighbouring bands are small. Calling the smallest of them the best band is only
meaningful if a training run reproduces itself more tightly than that, and nothing in a single
sweep says whether it does: one run per band cannot separate a band that is genuinely better from
a band that drew a better seed.

So the same band is trained several times with different seeds and nothing else changed. The
spread that comes back is the resolution of the whole mode study, and the knee is then defined
against it -- the narrowest band whose error is within the noise of the best -- rather than
against a tolerance picked to look reasonable.

Two bands per dataset, one on the steep part and one on the flat, because there is no reason to
assume the spread is the same in both regimes and the flat one is where it matters.
"""
import csv
import os
import statistics
import sys

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.fno.data import load_burgers, load_darcy
from torch_fno.fno.train import train

ROOT = os.environ.get("FNO_DATA", "/home/anirudh/fno/data")
OUT = os.environ.get("MODE_NOISE_OUT", "/home/anirudh/fno/mode_noise_gpu.csv")
EPOCHS = int(os.environ.get("MODE_ACC_EPOCHS", 30))
SEEDS = [0, 1, 2, 3]
FIELDS = ["case", "n_modes", "seed", "rel_h1"]

# Navier--Stokes is left out: one run is a third of an hour, so four seeds at two bands would
# cost more than the sweep they are there to calibrate. The spread is reported from the three
# datasets that can afford it and applied to all four, which is an assumption and is stated.
CASES = [("Burgers", lambda: load_burgers(ROOT), 24, 16, [6, 14]),
         ("Darcy-32", lambda: load_darcy(ROOT, 32), 24, 16, [8, 24]),
         ("Darcy-64", lambda: load_darcy(ROOT, 64), 24, 16, [12, 32])]


def main():
    rows = []
    for name, loader, channels, batch, bands in CASES:
        xtr, ytr, xte, yte = loader()
        for modes in bands:
            errors = []
            for seed in SEEDS:
                _, best_h1, _ = train(xtr, ytr, xte, yte, n_modes=modes, channels=channels,
                                      epochs=EPOCHS, batch=batch, loss="h1", seed=seed,
                                      device="cuda", verbose=False)
                errors.append(best_h1)
                rows.append({"case": name, "n_modes": modes, "seed": seed,
                             "rel_h1": f"{best_h1:.6f}"})
                torch.cuda.empty_cache()
            lo, hi = min(errors), max(errors)
            print(f"{name:9} m={modes:3d}  "
                  f"mean {statistics.mean(errors):.4f}  sd {statistics.stdev(errors):.4f}  "
                  f"spread {hi / lo:.4f}x", flush=True)

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
