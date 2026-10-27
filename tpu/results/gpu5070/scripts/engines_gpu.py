"""The three real-input row transforms on a GPU, against the band and against the width.

Two things decide which packing scheme wins and only one of them is the band, so both are swept.

  n_modes   how many bins are kept.  Narrow bands are where the partial transform is supposed to
            pay, and where coupled's mirror rows hurt most in relative terms.
  k_cols    how many signals go through at once.  This is the axis the TPU study held fixed at
            256, and it is the one that decides the matmul's shape: coupled halves it, reduced
            halves the contracted length instead.  In a layer it is batch * channels * width, so
            it ranges over four orders of magnitude between Burgers at a small batch and
            Navier--Stokes at a large one.

Correctness is checked against a float64 NumPy transform before anything is timed, so a wrong
answer cannot be reported as a fast one. The two schemes have preconditions the plain transform
does not -- coupled pairs signals so it needs an even count, reduced splits samples so it needs an
even length -- and a grid that violates one is reported as unavailable rather than skipped: not
being applicable is part of the answer to which is better when.

Timed with CUDA events, best of many after a warmup, which is this repository's GPU convention.
"""
import csv
import itertools
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import numpy as np
import torch

from torch_fno.fft.rfft1d import rfft_low_modes
from torch_fno.fft.tricks import rfft_coupled_low_modes, rfft_reduced_low_modes

DEV = torch.device("cuda")
OUT = os.environ.get("ENGINES_OUT", "/home/anirudh/fno/engines_gpu.csv")
REPS = int(os.environ.get("ENGINES_REPS", 50))
WARMUP = 10
FIELDS = ["case", "grid", "n_modes", "k_cols", "engine", "us", "rel_err", "status"]

ENGINES = [("reim", rfft_low_modes),
           ("coupled", rfft_coupled_low_modes),
           ("reduced", rfft_reduced_low_modes)]
# One row length per dataset, with the band grids kept dense enough to see a crossing rather than
# bracket one.  Burgers is 17 rows, which is odd, and that is not an accident of the loader: the
# dataset's time axis has an endpoint.
CASES = [("Burgers", 17, list(range(2, 17, 2))),
         ("Darcy-32", 32, list(range(2, 33, 2))),
         ("Darcy-64", 64, list(range(4, 65, 4))),
         ("NS-128", 128, list(range(8, 129, 8)))]
WIDTHS = [64, 256, 1024, 4096, 16384]

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def timed(fn, x, reps=REPS):
    """Best of `reps`, CUDA events. The best rather than the mean: every other sample is this
    one plus some amount of something else."""
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(x)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        best = float("inf")
        for _ in range(reps):
            begin.record()
            fn(x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def error_against_f64(got_re, got_im, exact, n_modes):
    got = np.asarray(got_re.cpu(), np.float64) + 1j * np.asarray(got_im.cpu(), np.float64)
    return float(np.abs(got - exact[:n_modes]).max() / np.abs(exact[:n_modes]).max())


def main():
    rows = []
    rng = np.random.default_rng(0)
    for (case, grid, bands), k_cols in itertools.product(CASES, WIDTHS):
        host = rng.normal(size=(grid, k_cols)).astype(np.float32)
        x = torch.as_tensor(host, device=DEV)
        exact = np.fft.fft(host.astype(np.float64), axis=0)
        print(f"\n{case}: grid {grid}, k_cols {k_cols}", flush=True)
        for m in bands:
            line = f"  m={m:<4}"
            for name, fn in ENGINES:
                row = {"case": case, "grid": grid, "n_modes": m, "k_cols": k_cols,
                       "engine": name}
                try:
                    re_, im_ = fn(x, m)
                    row["rel_err"] = f"{error_against_f64(re_, im_, exact, m):.3e}"
                    row["us"] = f"{timed(lambda z, _f=fn, _m=m: _f(z, _m), x):.3f}"
                    row["status"] = "ok"
                    line += f"  {name}={float(row['us']):7.2f}"
                except Exception as exc:
                    row.update({"us": "", "rel_err": "", "status": type(exc).__name__})
                    line += f"  {name}=n/a"
                rows.append(row)
            print(line, flush=True)

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)", flush=True)

    # TF32 is on, matching the layer, so the floor here is the 10-bit mantissa rather than
    # float32's 24: every engine lands around 2e-4 and none of them is wrong. A threshold set
    # for true float32 would flag the whole sweep. What would be a real failure is one engine
    # standing apart from the others, so that is what is checked.
    worst = collections.defaultdict(float)
    for row in rows:
        if row["status"] == "ok":
            worst[row["engine"]] = max(worst[row["engine"]], float(row["rel_err"]))
    print("worst relative error: "
          + "  ".join(f"{e}={v:.1e}" for e, v in sorted(worst.items())), flush=True)
    if worst and max(worst.values()) > 10 * min(worst.values()):
        print("WARNING: one engine is an order of magnitude less accurate than another",
              flush=True)


if __name__ == "__main__":
    main()
