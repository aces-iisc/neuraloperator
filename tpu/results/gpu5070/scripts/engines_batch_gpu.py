"""The three real-input transforms over a (B, N, num_modes) volume, in 1-D and in 2-D.

The companion grid sweep held the element count fixed, which answers "which engine wins when
every grid is given enough work to fill the machine".  This one holds the BATCH fixed instead
and sweeps it, which answers the different question of what each engine costs at a batch the
caller actually has.  The two are not interchangeable: a fixed batch means the work per cell
grows with N, so the small-N end sits on the launch floor and the large-N end does not.

B is the batch in both dimensionalities -- the number of independent signals in 1-D, the number
of fields in 2-D.  The data is therefore B*N floats in 1-D and B*N^2 in 2-D, so the 2-D volume
is a wedge rather than a box: one 4096^2 field is 67 MiB on its own, and the row pass needs the
field and its permuted copy at once.  Cells past the budget are recorded as skipped, not
dropped, so the hole in the volume is visible rather than implied.

Each cell is timed twice: eager, which is what the code does today, and a captured CUDA graph,
which leaves the device work.  Correctness is checked against a float64 NumPy transform on a
narrow input before timing.
"""
import csv
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
OUT = os.environ.get("BATCH_OUT", "/home/anirudh/fno/engines_batch_gpu.csv")
REPS = int(os.environ.get("BATCH_REPS", 15))
WARMUP = 5
POINTS = int(os.environ.get("BATCH_POINTS", 8))
# Input bytes one cell may ask for before it is skipped.  The row pass holds the field and its
# permuted copy at the same time, so the live figure is about twice this.
BUDGET = float(os.environ.get("BATCH_BUDGET", 3.0)) * 2 ** 30
CHECK_COLS = 32
CHECK_FIELDS = 2
FIELDS = ["dims", "family", "grid", "B", "n_modes", "engine",
          "eager_us", "graph_us", "rel_err", "status"]

GRIDS = [2 ** p for p in range(4, 13)]
BATCHES = [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
BAND = [("reim", rfft_low_modes),
        ("coupled", rfft_coupled_low_modes),
        ("reduced", rfft_reduced_low_modes)]
STAGED = [("reim", rfft_half), ("coupled", rfft_coupled_half),
          ("reduced", rfft_reduced_half)]
FFT2 = FFT(dtype=F)


def modes_grid(grid, points=POINTS):
    """About `points` values of num_modes from 2 to N, even and log-spaced."""
    raw = np.unique(np.round(np.geomspace(2, grid, points) / 2).astype(int) * 2)
    return [int(m) for m in raw if 2 <= m <= grid]


def input_bytes(dims, grid, batch):
    return batch * grid * 4 if dims == 1 else batch * grid * grid * 4


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
        row.update({"eager_us": "", "graph_us": "",
                    "status": "OOM" if "out of memory" in str(exc).lower()
                    else type(exc).__name__})
        torch.cuda.empty_cache()
        return f"{row['engine']}:{row['status'][:10]}"
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
    gen = torch.Generator(device=DEV).manual_seed(0)
    rows_out = []
    blank = {f: "" for f in FIELDS}
    print(f"radix {RADIX}, {len(GRIDS)} grids x {len(BATCHES)} batches, "
          f"budget {BUDGET / 2 ** 30:.1f} GiB of input per cell", flush=True)

    for grid in GRIDS:
        modes = modes_grid(grid)
        for batch in BATCHES:
            for dims in (1, 2):
                want = input_bytes(dims, grid, batch)
                if want > BUDGET:
                    for n_modes in modes:
                        for name in [e for e, _ in BAND] + ["native"]:
                            rows_out.append(dict(blank, dims=dims, family="band", grid=grid,
                                                 B=batch, n_modes=n_modes, engine=name,
                                                 status="skip:mem"))
                    print(f"  {dims}-D N={grid:<5} B={batch:<6} skipped, "
                          f"{want / 2 ** 30:.1f} GiB of input", flush=True)
                    continue
                # Built on the device.  At B=20000 the 2-D field set is gigabytes, and
                # generating it on the host only to copy it across costs more than the sweep.
                # The float64 check wants host data, so only the narrow slice it reads comes back.
                if dims == 1:
                    x = torch.randn((grid, batch), device=DEV, generator=gen, dtype=torch.float32)
                    check = x[:, :min(CHECK_COLS, batch)].cpu().numpy()
                else:
                    x = torch.randn((batch, 1, grid, grid), device=DEV, generator=gen,
                                    dtype=torch.float32)
                    check = x[:min(CHECK_FIELDS, batch)].cpu().numpy()
                print(f"\n  {dims}-D N={grid}  B={batch}  "
                      f"{want / 2 ** 20:.1f} MiB  {len(modes)} bands", flush=True)
                for n_modes in modes:
                    line = []
                    for name, engine in BAND:
                        row = dict(blank, dims=dims, family="band", grid=grid, B=batch,
                                   n_modes=n_modes, engine=name)
                        if dims == 1:
                            run = lambda e=engine, m=n_modes: e(x, m)
                            chk = lambda e=engine, m=n_modes: err_1d(e, check, m)
                        else:
                            run = lambda e=engine, m=n_modes: two_d(e, x, m)
                            chk = lambda e=engine, m=n_modes: err_2d(e, check, m)
                        line.append(measure(row, run, chk))
                        rows_out.append(row)
                    row = dict(blank, dims=dims, family="band", grid=grid, B=batch,
                               n_modes=n_modes, engine="native")
                    base = native_1d if dims == 1 else native_2d
                    line.append(measure(row, lambda m=n_modes, f=base: f(x, m)))
                    rows_out.append(row)
                    print(f"    m={n_modes:<5} " + "  ".join(line), flush=True)
                if dims == 1:
                    line = []
                    for name, engine in STAGED:
                        row = dict(blank, dims=1, family="staged", grid=grid, B=batch,
                                   n_modes=grid // 2, engine=name)
                        line.append(measure(row, lambda e=engine: e(x),
                                            lambda e=engine: err_1d(lambda z, _m: e(z),
                                                                    check, grid // 2)))
                        rows_out.append(row)
                    print("    staged  " + "  ".join(line), flush=True)
                del x
                core._rows_matrix.cache_clear()
                core._angle_key.cache_clear()
                torch.cuda.empty_cache()
                flush(rows_out)

    flush(rows_out)
    ok = [r for r in rows_out if r["status"] == "ok"]
    skipped = [r for r in rows_out if r["status"] == "skip:mem"]
    bad = [r for r in rows_out if r["rel_err"]]
    print(f"\nwrote {OUT}: {len(rows_out)} rows, {len(ok)} ok, {len(skipped)} over budget, "
          f"{len(rows_out) - len(ok) - len(skipped)} failed", flush=True)
    if bad:
        worst = max(bad, key=lambda r: float(r["rel_err"]))
        print(f"worst rel_err {worst['rel_err']} at {worst['dims']}-D N={worst['grid']} "
              f"B={worst['B']} m={worst['n_modes']} {worst['engine']}", flush=True)
    print("BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
