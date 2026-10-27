"""Which real-input transform wins, as a function of the grid and the band.

Every engine comparison so far fixed one of the two axes. The band bench fixed a narrow
num_modes and swept the width; the full-spectrum table fixed num_modes at the whole spectrum
and swept N. Each found a different winner, which is only a contradiction if the two axes are
assumed independent. They are not, so this sweeps both: N from 16 to 4096 in powers of two,
num_modes from 2 to N in powers of two, all three engines at every cell.

Two families run, because "the same engine" means two different programs:

  band    one direct partial DFT, the matrix restricted to the rows asked for. This is what the
          layer runs. All three are flop-equivalent here at num_modes = N/2 -- two m x N/2
          matrices hold the same entries as one m x N matrix -- so a packing scheme can only win
          by changing the matmul's shape, never by doing less arithmetic.
  staged  a radix-staged transform producing the whole non-redundant half. Here the length sets
          the cost, because it decides how many stages the plan needs, and halving it can delete
          a stage rather than shrink a matrix. That is a saving no band version can reach, and it
          is the family the published full-spectrum table measured.

Each cell is timed twice. Eager is what the code does today: one kernel launch per operation,
dispatched from Python. Below a few hundred points that dominates, and the comparison degenerates
into a count of how many operations each scheme writes -- a property of this binding, not of the
hardware or the algorithm. Replaying a captured CUDA graph removes the launch path and leaves the
device work, which is what the TPU numbers measure, since XLA compiles the whole engine into one
program. Both columns are kept: the first is what a user gets, the second is what the scheme is
worth.

`torch.fft.rfft` runs alongside as the library reference. It cannot truncate, so it computes the
whole spectrum at every cell and is sliced afterwards -- which is the point of the comparison.

Correctness is checked against a float64 NumPy transform on a narrow input before anything is
timed; the timed shapes are too large to check cell by cell.
"""
import csv
import itertools
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import numpy as np
import torch

from torch_fno.fft import core
from torch_fno.fft.core import RADIX
from torch_fno.fft.rfft1d import rfft_half, rfft_low_modes
from torch_fno.fft.tricks import (rfft_coupled_half, rfft_coupled_low_modes,
                                  rfft_reduced_half, rfft_reduced_low_modes)

DEV = torch.device("cuda")
OUT = os.environ.get("PHASE_OUT", "/home/anirudh/fno/engines_phase_gpu.csv")
REPS = int(os.environ.get("PHASE_REPS", 30))
WARMUP = 8
K_CHECK = 32
FIELDS = ["family", "grid", "k_cols", "n_modes", "engine", "eager_us", "graph_us",
          "rel_err", "status"]

GRIDS = [2 ** p for p in range(4, 13)]
WIDTHS = [int(k) for k in os.environ.get("PHASE_K", "256,1024").split(",")]

BAND = [("reim", rfft_low_modes),
        ("coupled", rfft_coupled_low_modes),
        ("reduced", rfft_reduced_low_modes)]
STAGED = [("reim", rfft_half),
          ("coupled", rfft_coupled_half),
          ("reduced", rfft_reduced_half)]


def bands(grid):
    """num_modes in powers of two from 2 up to N, so the last cell is the whole spectrum."""
    return [2 ** p for p in range(1, int(grid).bit_length())]


def _best_of(step, reps):
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(reps):
        begin.record()
        step()
        end.record()
        torch.cuda.synchronize()
        best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def timed_eager(fn, reps=REPS):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        return _best_of(fn, reps)


def timed_graph(fn, reps=REPS):
    """Capture one call and time the replay, so the launch path is out of the measurement."""
    with torch.no_grad():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(WARMUP):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        best = _best_of(graph.replay, reps)
    del graph
    torch.cuda.synchronize()
    return best


def error(got_re, got_im, host, keep):
    got = np.asarray(got_re.cpu(), np.float64) + 1j * np.asarray(got_im.cpu(), np.float64)
    exact = np.fft.fft(host.astype(np.float64), axis=0)[:keep]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def native_band(x, n_modes):
    """`torch.fft` with the band applied afterwards, since it cannot be applied before."""
    return torch.fft.rfft(x, dim=0)[:n_modes]


def measure(row, run, check=None):
    """Fill `row` with both timings, and the error if a check was supplied."""
    try:
        if check is not None:
            row["rel_err"] = f"{check():.3e}"
        row["eager_us"] = f"{timed_eager(run):.4f}"
        row["status"] = "ok"
    except Exception as exc:
        row.update({"eager_us": "", "graph_us": "", "status": type(exc).__name__})
        return f"{row['engine']} {row['status']}"
    try:
        row["graph_us"] = f"{timed_graph(run):.4f}"
    except Exception as exc:
        row["graph_us"] = ""
        row["status"] = f"ok/nograph:{type(exc).__name__}"
    return (f"{row['engine']} {float(row['eager_us']):8.2f}"
            + (f"/{float(row['graph_us']):<8.2f}" if row["graph_us"] else "/--      "))


def main():
    rng = np.random.default_rng(0)
    rows_out = []

    for grid, k_cols in itertools.product(GRIDS, WIDTHS):
        host = rng.normal(size=(grid, k_cols)).astype(np.float32)
        x = torch.as_tensor(host, device=DEV)
        check_host = host[:, :K_CHECK]
        check_x = torch.as_tensor(check_host, device=DEV)
        print(f"\nN={grid}  K={k_cols}   (eager/graph us)", flush=True)

        for n_modes in bands(grid):
            line = []
            for name, engine in BAND:
                row = {"family": "band", "grid": grid, "k_cols": k_cols,
                       "n_modes": n_modes, "engine": name, "rel_err": "", "graph_us": ""}
                line.append(measure(
                    row, lambda e=engine, m=n_modes: e(x, m),
                    lambda e=engine, m=n_modes: error(*e(check_x, m), check_host, m)))
                rows_out.append(row)

            row = {"family": "band", "grid": grid, "k_cols": k_cols,
                   "n_modes": n_modes, "engine": "native", "rel_err": "", "graph_us": ""}
            line.append(measure(row, lambda m=n_modes: native_band(x, m)))
            rows_out.append(row)
            print(f"  m={n_modes:<5} " + "  ".join(line), flush=True)

        line = []
        for name, engine in STAGED:
            row = {"family": "staged", "grid": grid, "k_cols": k_cols,
                   "n_modes": grid // 2, "engine": name, "rel_err": "", "graph_us": ""}
            line.append(measure(
                row, lambda e=engine: e(x),
                lambda e=engine: error(*e(check_x), check_host, grid // 2)))
            rows_out.append(row)
        print("  staged      " + "  ".join(line), flush=True)

        # The row matrices are the only thing here that grows without bound, and at N = 4096 a
        # single coupled matrix is a quarter of a gigabyte.
        core._rows_matrix.cache_clear()
        core._angle_key.cache_clear()
        del x, check_x
        torch.cuda.empty_cache()

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    worst = max((r for r in rows_out if r["rel_err"]), key=lambda r: float(r["rel_err"]))
    print(f"\nradix {RADIX}, wrote {OUT}  ({len(rows_out)} rows)", flush=True)
    print(f"worst rel_err {worst['rel_err']} at {worst['family']} N={worst['grid']} "
          f"m={worst['n_modes']} {worst['engine']}", flush=True)


if __name__ == "__main__":
    main()
