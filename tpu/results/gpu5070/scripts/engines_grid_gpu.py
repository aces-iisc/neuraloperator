"""The three real-input transforms over a dense (grid, num_modes) plane, in 1-D and in 2-D.

The earlier phase sweep took num_modes in powers of two, which is eight points at most and
leaves the boundary between two engines somewhere inside a factor of two. This takes about two
dozen values of num_modes per grid, so a boundary can be located rather than bracketed, and it
runs the two-dimensional transform as well as the one-dimensional one.

The two dimensionalities are not the same experiment. In 1-D the engine is the whole transform.
In 2-D only the first pass sees real data -- by the second pass the field is complex -- so the
engine reaches the row pass and the column pass is the same work whichever engine is chosen.
The 2-D columns therefore measure how much of a separable transform the choice can touch at all,
which is the number that decides whether it is worth choosing.

Two conventions, both forced by what the engines return:

  * all three produce the LOWEST num_modes row bins. That is not the band an FNO layer keeps --
    the row axis is not conjugate-symmetric, so the layer takes both low corners -- but it is
    what these three compute, and comparing them means asking all three for the same thing.
  * the field count falls as the grid grows, holding the element count near fixed. A 4096 x 4096
    field is 67 MB on its own; a batch of them is not a measurement, it is an allocation failure.

Each cell is timed twice: eager, which is what the code does today and below a few hundred
points is dominated by the launch path, and a captured CUDA graph, which leaves the device work.

Correctness is checked against a float64 NumPy transform on a narrow input before timing.
"""
import csv
import itertools
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import numpy as np
import torch

from torch_fno.fft import core
from torch_fno.fft.core import F, RADIX
from torch_fno.fft.rfft1d import rfft_half, rfft_low_modes
from torch_fno.fft.transform import FFT
from torch_fno.fft.tricks import (rfft_coupled_half, rfft_coupled_low_modes,
                                  rfft_reduced_half, rfft_reduced_low_modes)

DEV = torch.device("cuda")
OUT = os.environ.get("GRID_OUT", "/home/anirudh/fno/engines_grid_gpu.csv")
REPS = int(os.environ.get("GRID_REPS", 20))
WARMUP = 6
POINTS = int(os.environ.get("GRID_POINTS", 24))
ELEMENTS = int(os.environ.get("GRID_ELEMENTS", 1 << 24))
K_CAP = int(os.environ.get("GRID_K_CAP", 1 << 18))
CHECK_FIELDS = 2
FIELDS = ["dims", "family", "grid", "k_cols", "fields", "n_modes", "engine",
          "eager_us", "graph_us", "rel_err", "status"]

GRIDS = [2 ** p for p in range(4, 13)]
BAND = [("reim", rfft_low_modes),
        ("coupled", rfft_coupled_low_modes),
        ("reduced", rfft_reduced_low_modes)]
STAGED = [("reim", rfft_half), ("coupled", rfft_coupled_half),
          ("reduced", rfft_reduced_half)]
FFT2 = FFT(dtype=F)


def modes_grid(grid, points=POINTS):
    """About `points` values of num_modes from 2 to N, even and log-spaced.

    Log rather than linear because every cost here is a ratio: the interesting structure sits
    near the narrow end, where an FNO lives, and a linear grid spends most of its points out on
    the flat.
    """
    raw = np.unique(np.round(np.geomspace(2, grid, points) / 2).astype(int) * 2)
    return [int(m) for m in raw if 2 <= m <= grid]


def width_for(grid):
    """Signals per 1-D cell, holding the element count fixed so every grid is compute-bound.

    A width fixed across grids measures the launch path at the small end, where the matmul is
    too short to fill the unit: the time goes flat and the map becomes a kernel count. Holding
    the element count instead puts every grid on the same side of that floor. The cap keeps the
    smallest grids from going memory-bound at the other extreme.
    """
    return int(min(K_CAP, max(256, ELEMENTS // grid)))


def fields_for(grid):
    """Fields per 2-D cell, on the same element budget and capped at a sane batch."""
    return int(min(256, max(1, ELEMENTS // (grid * grid))))


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


# -- the two transforms ----------------------------------------------------------------
def row_pass(engine, x, n_modes):
    """(n, c, h, w) real -> (re, im) each (n, c, n_modes, w), the engine doing the row axis."""
    n, c, h, w = x.shape
    flat = x.permute(2, 0, 1, 3).reshape(h, n * c * w)
    re, im = engine(flat, n_modes)
    rows = re.shape[0]
    return (re.reshape(rows, n, c, w).permute(1, 2, 0, 3),
            im.reshape(rows, n, c, w).permute(1, 2, 0, 3))


def two_d(engine, x, n_modes):
    """The whole separable transform: engine on the real rows, complex pass on the columns."""
    re, im = row_pass(engine, x, n_modes)
    return FFT2.cols_forward(re, im, FFT2.half_band(n_modes), x.shape[-1])


def native_1d(x, n_modes):
    return torch.fft.rfft(x, dim=0)[:n_modes]


def native_2d(x, n_modes):
    return torch.fft.rfft2(x, dim=(-2, -1))[:, :, :n_modes, :max(1, n_modes // 2)]


# -- correctness -----------------------------------------------------------------------
def err_1d(engine, host, n_modes):
    re, im = engine(torch.as_tensor(host, device=DEV), n_modes)
    got = np.asarray(re.cpu(), np.float64) + 1j * np.asarray(im.cpu(), np.float64)
    exact = np.fft.fft(host.astype(np.float64), axis=0)[:n_modes]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def err_2d(engine, host, n_modes):
    re, im = two_d(engine, torch.as_tensor(host, device=DEV), n_modes)
    got = np.asarray(re.cpu(), np.float64) + 1j * np.asarray(im.cpu(), np.float64)
    full = np.fft.fft2(host.astype(np.float64), axes=(-2, -1))
    cols = [int(c) for c in FFT2.half_band(n_modes)]
    exact = full[:, :, :n_modes][:, :, :, cols]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def measure(row, run, check=None):
    try:
        if check is not None:
            row["rel_err"] = f"{check():.3e}"
        row["eager_us"] = f"{timed_eager(run):.4f}"
        row["status"] = "ok"
    except Exception as exc:
        row.update({"eager_us": "", "graph_us": "", "status": type(exc).__name__})
        torch.cuda.empty_cache()
        return f"{row['engine']}:{row['status'][:12]}"
    try:
        row["graph_us"] = f"{timed_graph(run):.4f}"
    except Exception as exc:
        row["graph_us"] = ""
        row["status"] = f"ok/nograph:{type(exc).__name__}"
    return f"{row['engine']} {float(row['eager_us']):8.1f}"


def flush(rows_out):
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)


def main():
    rng = np.random.default_rng(0)
    rows_out = []
    blank = {f: "" for f in FIELDS}

    for grid in GRIDS:
        modes = modes_grid(grid)
        fields = fields_for(grid)
        width = width_for(grid)
        host1 = rng.normal(size=(grid, width)).astype(np.float32)
        x1 = torch.as_tensor(host1, device=DEV)
        host2 = rng.normal(size=(fields, 1, grid, grid)).astype(np.float32)
        x2 = torch.as_tensor(host2, device=DEV)
        check1 = host1[:, :32]
        check2 = host2[:CHECK_FIELDS]
        print(f"\nN={grid}  1-D K={width}  2-D fields={fields}  "
              f"{len(modes)} values of num_modes", flush=True)

        for n_modes in modes:
            for dims, engines, native, checker in (
                    (1, BAND, native_1d, err_1d), (2, BAND, native_2d, err_2d)):
                line = []
                for name, engine in engines:
                    row = dict(blank, dims=dims, family="band", grid=grid,
                               k_cols=width if dims == 1 else "", fields="" if dims == 1
                               else fields, n_modes=n_modes, engine=name)
                    if dims == 1:
                        run = lambda e=engine, m=n_modes: e(x1, m)
                        chk = lambda e=engine, m=n_modes: checker(e, check1, m)
                    else:
                        run = lambda e=engine, m=n_modes: two_d(e, x2, m)
                        chk = lambda e=engine, m=n_modes: checker(e, check2, m)
                    line.append(measure(row, run, chk))
                    rows_out.append(row)

                row = dict(blank, dims=dims, family="band", grid=grid,
                           k_cols=width if dims == 1 else "", fields="" if dims == 1
                           else fields, n_modes=n_modes, engine="native")
                target = x1 if dims == 1 else x2
                line.append(measure(row, lambda m=n_modes, t=target, f=native: f(t, m)))
                rows_out.append(row)
                print(f"  {dims}-D m={n_modes:<5} " + "  ".join(line), flush=True)

        line = []
        for name, engine in STAGED:
            row = dict(blank, dims=1, family="staged", grid=grid, k_cols=width,
                       fields="", n_modes=grid // 2, engine=name)
            line.append(measure(row, lambda e=engine: e(x1),
                                lambda e=engine: err_1d(lambda z, _m: e(z),
                                                        check1, grid // 2)))
            rows_out.append(row)
        print("  1-D staged   " + "  ".join(line), flush=True)

        flush(rows_out)
        core._rows_matrix.cache_clear()
        core._angle_key.cache_clear()
        del x1, x2
        torch.cuda.empty_cache()

    flush(rows_out)
    bad = [r for r in rows_out if r["rel_err"]]
    worst = max(bad, key=lambda r: float(r["rel_err"])) if bad else None
    print(f"\nradix {RADIX}, wrote {OUT}  ({len(rows_out)} rows)", flush=True)
    if worst:
        print(f"worst rel_err {worst['rel_err']} at {worst['dims']}-D N={worst['grid']} "
              f"m={worst['n_modes']} {worst['engine']}", flush=True)
    failed = [r for r in rows_out if r["status"] not in ("ok",) and not
              r["status"].startswith("ok/")]
    print(f"{len(failed)} cells failed", flush=True)
    for r in failed[:12]:
        print(f"   {r['dims']}-D N={r['grid']} m={r['n_modes']} {r['engine']}: {r['status']}",
              flush=True)
    print("GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
