"""The three real-input transforms across the whole range of num_modes, in 1-D and in 2-D.

The earlier engine bench stopped where an operator stops, at a narrow band. That is the regime
the layer runs in, but it is not the regime the schemes were designed for, and a method that
loses everywhere it was measured may simply never have been measured where it wins. So this
sweeps num_modes from a handful up to N, the full spectrum, on both the one-dimensional row
transform and a two-dimensional transform built from them.

One structural fact shapes the 2-D half and is worth stating before the numbers. Both packing
schemes exploit the input being real: coupled pairs two real signals into one complex signal,
reduced splits a real signal into its even and odd samples. In a separable 2-D transform only
the first pass sees real data; by the second pass the field is complex. The engine choice
therefore reaches the row pass only, and the column pass is the same complex transform whichever
engine is chosen. The 2-D columns here measure the whole two-pass transform, so they show what
fraction of it the choice can touch at all.

Correctness is checked against a float64 NumPy transform before anything is timed.
"""
import csv
import itertools
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import numpy as np
import torch

from torch_fno.fft.core import ACC, F, stacked_rows
from torch_fno.fft.rfft1d import rfft_low_modes
from torch_fno.fft.transform import FFT
from torch_fno.fft.tricks import rfft_coupled_low_modes, rfft_reduced_low_modes

DEV = torch.device("cuda")
OUT = os.environ.get("ENGINES_FULL_OUT", "/home/anirudh/fno/engines_full_gpu.csv")
REPS = int(os.environ.get("ENGINES_REPS", 40))
WARMUP = 10
FIELDS = ["dims", "grid", "n_modes", "k_cols", "engine", "us", "rel_err", "status"]

ENGINES = [("reim", rfft_low_modes),
           ("coupled", rfft_coupled_low_modes),
           ("reduced", rfft_reduced_low_modes)]
# Square grids only, so "num_modes up to N" means the same thing on both axes. 17 is left out
# because the even/odd split cannot run on it and the point here is to compare all three.
GRIDS = [32, 64, 128]
K_1D = int(os.environ.get("ENGINES_K", 4096))
BATCH_2D, CHANNELS_2D = 8, 24

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def bands(grid, count=16):
    """num_modes from 2 to N inclusive, so the full spectrum is the last point."""
    step = max(2, 2 * round(grid / count / 2))
    out = sorted({m for m in range(step, grid + 1, step)} | {2, grid})
    return [m for m in out if m >= 2]


def timed(fn, reps=REPS):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        best = float("inf")
        for _ in range(reps):
            begin.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def row_pass(engine, x, n_modes):
    """(n, c, h, w) real -> (re, im) each (n, c, rows, w), with `engine` doing the transform.

    The engine works on a real (N, K) block, so the grid rows move to the front and everything
    else folds into the column axis. That reshape is what the layer does anyway.
    """
    n, c, h, w = x.shape
    flat = x.permute(2, 0, 1, 3).reshape(h, n * c * w)
    re, im = engine(flat, n_modes)
    rows = re.shape[0]
    return (re.reshape(rows, n, c, w).permute(1, 2, 0, 3),
            im.reshape(rows, n, c, w).permute(1, 2, 0, 3))


def two_d(engine, x, n_modes, fft):
    """The whole separable transform: engine on the real row axis, complex pass on the columns."""
    re, im = row_pass(engine, x, n_modes)
    cols = fft.half_band(n_modes)
    return fft.cols_forward(re, im, cols, x.shape[-1])


def check_1d(engine, host, n_modes):
    x = torch.as_tensor(host, device=DEV)
    re, im = engine(x, n_modes)
    got = np.asarray(re.cpu(), np.float64) + 1j * np.asarray(im.cpu(), np.float64)
    exact = np.fft.fft(host.astype(np.float64), axis=0)[:n_modes]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def check_2d(engine, host, n_modes, fft):
    x = torch.as_tensor(host, device=DEV)
    re, im = two_d(engine, x, n_modes, fft)
    got = np.asarray(re.cpu(), np.float64) + 1j * np.asarray(im.cpu(), np.float64)
    full = np.fft.fft2(host.astype(np.float64), axes=(-2, -1))
    # The engines all return the LOWEST n_modes row bins. That is not the band an FNO layer
    # keeps -- the row axis is not conjugate-symmetric, so the layer takes both low corners --
    # but it is what these three compute, and comparing them requires asking all three for the
    # same thing. The column axis is one-sided for all of them and does match the layer.
    cols = [int(c) for c in fft.half_band(n_modes)]
    exact = full[:, :, :n_modes][:, :, :, cols]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def main():
    rng = np.random.default_rng(0)
    fft = FFT(dtype=F)
    rows_out = []

    for grid, (name, engine) in itertools.product(GRIDS, ENGINES):
        host1 = rng.normal(size=(grid, K_1D)).astype(np.float32)
        x1 = torch.as_tensor(host1, device=DEV)
        host2 = rng.normal(size=(BATCH_2D, CHANNELS_2D, grid, grid)).astype(np.float32)
        x2 = torch.as_tensor(host2, device=DEV)
        print(f"\ngrid {grid}, {name}", flush=True)

        for dims, label in ((1, "1-D"), (2, "2-D")):
            for m in bands(grid):
                row = {"dims": dims, "grid": grid, "n_modes": m,
                       "k_cols": K_1D if dims == 1 else BATCH_2D * CHANNELS_2D * grid,
                       "engine": name}
                try:
                    if dims == 1:
                        row["rel_err"] = f"{check_1d(engine, host1, m):.3e}"
                        row["us"] = f"{timed(lambda: engine(x1, m)):.3f}"
                    else:
                        row["rel_err"] = f"{check_2d(engine, host2, m, fft):.3e}"
                        row["us"] = f"{timed(lambda: two_d(engine, x2, m, fft)):.3f}"
                    row["status"] = "ok"
                    print(f"  {label} m={m:<4} {float(row['us']):9.2f} us  "
                          f"err {row['rel_err']}", flush=True)
                except Exception as exc:
                    row.update({"us": "", "rel_err": "", "status": type(exc).__name__})
                    print(f"  {label} m={m:<4} {row['status']}: {str(exc)[:60]}", flush=True)
                rows_out.append(row)

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {OUT}  ({len(rows_out)} rows)", flush=True)


if __name__ == "__main__":
    main()
