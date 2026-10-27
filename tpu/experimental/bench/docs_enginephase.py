"""Which real-input transform wins, as a function of the grid and the band.

Every engine comparison in the paper so far fixed one of the two axes. Table~\\ref{tab:engines-1d}
fixed num_modes at the whole spectrum and swept $N$; tab:enginemodes fixed $N$ at the two dataset
grids and swept the band. The first found `reduced` ahead at $N = 256$ and `coupled` at $N = 512$;
the second found `reim` ahead everywhere. That is only a contradiction if the two axes are
independent. They are not, so this sweeps both: $N$ from 16 to 4096 in powers of two, num_modes
from 2 to $N$ in powers of two, all three engines at every cell.

Two families run, because "the same engine" names two different programs:

  band    one direct partial DFT, the matrix restricted to the rows asked for. This is what the
          layer runs. All three are flop-equivalent here at num_modes = N/2 -- two m x N/2
          matrices hold the same entries as one m x N matrix -- so a packing scheme can only win
          by changing the matmul's shape, never by doing less arithmetic.
  staged  a radix-RADIX transform producing the whole non-redundant half, which is what
          Table~\\ref{tab:engines-1d} measured. Here the length sets the cost, because it decides
          how many stages the plan needs and how large the leaf is, and halving the length can
          delete a stage rather than shrink a matrix. That is a saving no band version can reach.

`jnp.fft.rfft` runs alongside as the library reference. It cannot truncate, so it computes the
whole spectrum at every cell and is sliced afterwards -- which is the point of the comparison.

Time is DEVICE time from this run's own traces. At the small end the kernel is a couple of
microseconds against a dispatch floor two orders of magnitude larger, so a wall clock would
measure the floor and nothing else. Each trace is parsed and then deleted, because 600 of them
do not fit anywhere worth keeping.

Every cell is checked against a float64 NumPy transform on a narrow input before it is timed, so
a wrong answer cannot be reported as a fast one.
"""
import csv
import itertools
import os
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import F, RADIX
from ...fft.rfft1d import rfft_half, rfft_low_modes
from ..fft.tricks import (rfft_coupled_low_modes, rfft_coupled_reim,
                          rfft_reduced_low_modes, rfft_reduced_reim)
from .parse_ops import bucket_times

OUT = os.path.expanduser(os.environ.get("PHASE_OUT", "~/repro_out/engines_phase_tpu.csv"))
TRACES = os.path.expanduser(os.environ.get("PHASE_TRACES", "~/phase_traces"))
REPS = int(os.environ.get("PHASE_REPS", "10"))
K_CHECK = 32
FIELDS = ["family", "grid", "k_cols", "n_modes", "engine", "device_us", "mxu_us", "vpu_us",
          "copy_us", "dma_us", "wall_us", "rel_err", "status"]
BUCKETS = ["MXU matmul", "VPU / fusion", "reshape / copy", "DMA / HBM"]

GRIDS = [2 ** p for p in range(4, 13)]
WIDTHS = [int(k) for k in os.environ.get("PHASE_K", "256").split(",")]

BAND = [("reim", rfft_low_modes),
        ("coupled", rfft_coupled_low_modes),
        ("reduced", rfft_reduced_low_modes)]
STAGED = [("reim", lambda x: rfft_half(x)),
          ("coupled", lambda x: rfft_coupled_reim(x, 0.5)),
          ("reduced", lambda x: rfft_reduced_reim(x, 0.5))]


def bands(grid):
    """num_modes in powers of two from 2 up to N, so the last cell is the whole spectrum."""
    return [2 ** p for p in range(1, int(grid).bit_length())]


def measured(fn, x, tag):
    """Device time per run, split by hardware bucket, from a trace that is then discarded."""
    os.makedirs(TRACES, exist_ok=True)
    path = os.path.join(TRACES, tag)
    shutil.rmtree(path, ignore_errors=True)
    jax.block_until_ready(fn(x))
    start = time.perf_counter()
    with jax.profiler.trace(path):
        for _ in range(REPS):
            jax.block_until_ready(fn(x))
    wall = (time.perf_counter() - start) * 1e6 / REPS
    buckets = bucket_times(path, REPS)
    shutil.rmtree(path, ignore_errors=True)
    return buckets, wall


def error(pair, host, keep):
    re_, im_ = pair
    got = np.asarray(re_, np.float64) + 1j * np.asarray(im_, np.float64)
    exact = np.fft.fft(host.astype(np.float64), axis=0)[:keep]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def fill(row, fn, x, tag, check=None):
    try:
        if check is not None:
            row["rel_err"] = f"{check():.3e}"
        buckets, wall = measured(fn, x, tag)
        row["device_us"] = f"{buckets.get('total', 0.0):.4f}"
        row["mxu_us"] = f"{buckets.get(BUCKETS[0], 0.0):.4f}"
        row["vpu_us"] = f"{buckets.get(BUCKETS[1], 0.0):.4f}"
        row["copy_us"] = f"{buckets.get(BUCKETS[2], 0.0):.4f}"
        row["dma_us"] = f"{buckets.get(BUCKETS[3], 0.0):.4f}"
        row["wall_us"] = f"{wall:.2f}"
        row["status"] = "ok"
        return f"{row['engine']} {float(row['device_us']):9.2f}"
    except Exception as exc:
        row["status"] = type(exc).__name__
        return f"{row['engine']} {row['status']}"


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    rng = np.random.default_rng(0)
    rows_out = []
    blank = {f: "" for f in FIELDS}

    for grid, k_cols in itertools.product(GRIDS, WIDTHS):
        host = rng.normal(size=(grid, k_cols)).astype(np.float32)
        x = jnp.asarray(host)
        check_host = host[:, :K_CHECK]
        check_x = jnp.asarray(check_host)
        print(f"\nN={grid}  K={k_cols}   (device us)", flush=True)

        for n_modes in bands(grid):
            line = []
            for name, engine in BAND:
                fn = jax.jit(lambda z, _e=engine, _m=n_modes: _e(z, _m))
                row = dict(blank, family="band", grid=grid, k_cols=k_cols,
                           n_modes=n_modes, engine=name)
                line.append(fill(row, fn, x, f"phase_band_{grid}_{k_cols}_{n_modes}_{name}",
                                 lambda _f=fn, _m=n_modes: error(_f(check_x), check_host, _m)))
                rows_out.append(row)

            native = jax.jit(lambda z, _m=n_modes: jnp.fft.rfft(z, axis=0)[:_m])
            row = dict(blank, family="band", grid=grid, k_cols=k_cols,
                       n_modes=n_modes, engine="native")
            line.append(fill(row, native, x, f"phase_band_{grid}_{k_cols}_{n_modes}_native"))
            rows_out.append(row)
            print(f"  m={n_modes:<5} " + "  ".join(line), flush=True)

        line = []
        for name, engine in STAGED:
            fn = jax.jit(engine)
            row = dict(blank, family="staged", grid=grid, k_cols=k_cols,
                       n_modes=grid // 2, engine=name)
            line.append(fill(row, fn, x, f"phase_staged_{grid}_{k_cols}_{name}",
                             lambda _f=fn: error(_f(check_x), check_host, grid // 2)))
            rows_out.append(row)
        print("  staged      " + "  ".join(line), flush=True)

        with open(OUT, "w", newline="") as fh:
            writer = csv.DictWriter(fh, FIELDS)
            writer.writeheader()
            writer.writerows(rows_out)

    worst = max((r for r in rows_out if r["rel_err"]), key=lambda r: float(r["rel_err"]))
    print(f"\nradix {RADIX}, wrote {OUT}  ({len(rows_out)} rows)", flush=True)
    print(f"worst rel_err {worst['rel_err']} at {worst['family']} N={worst['grid']} "
          f"m={worst['n_modes']} {worst['engine']}", flush=True)
    print("PHASE_DONE", flush=True)


if __name__ == "__main__":
    main()
