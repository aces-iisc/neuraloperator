"""The three real-input transforms over a dense (grid, num_modes) plane, in 1-D and in 2-D.

The companion to the GPU sweep of the same name, and the same experiment: about two dozen
values of num_modes per grid rather than the eight a powers-of-two ladder gives, so a boundary
between two engines can be located rather than bracketed, and the two-dimensional transform
measured beside the one-dimensional one.

The two dimensionalities are not the same experiment. In 1-D the engine is the whole transform.
In 2-D only the first pass sees real data -- by the second pass the field is complex -- so the
engine reaches the row pass and the column pass is the same work whichever engine is chosen.

Two conventions, both forced by what the engines return:

  * all three produce the LOWEST num_modes row bins. That is not the band an FNO layer keeps --
    the row axis is not conjugate-symmetric, so the layer takes both low corners -- but it is
    what these three compute, and comparing them means asking all three for the same thing.
  * the field count falls as the grid grows, holding the element count near fixed.

Time is DEVICE time from this run's own traces, because at the small end the kernel is a couple
of microseconds against a dispatch floor two orders of magnitude larger. Each trace is parsed
and then deleted. Every cell is checked against a float64 NumPy transform before it is timed.
"""
import csv
import os
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import F, RADIX
from ...fft.rfft1d import rfft_half, rfft_low_modes
from ...fft.transform import FFT
from ..fft.tricks import (rfft_coupled_low_modes, rfft_coupled_reim,
                          rfft_reduced_low_modes, rfft_reduced_reim)
from .parse_ops import bucket_times

OUT = os.path.expanduser(os.environ.get("GRID_OUT", "~/repro_out/engines_grid_tpu.csv"))
TRACES = os.path.expanduser(os.environ.get("GRID_TRACES", "~/grid_traces"))
REPS = int(os.environ.get("GRID_REPS", "8"))
POINTS = int(os.environ.get("GRID_POINTS", "24"))
ELEMENTS = int(os.environ.get("GRID_ELEMENTS", str(1 << 24)))
K_CAP = int(os.environ.get("GRID_K_CAP", str(1 << 18)))
CHECK_FIELDS = 2
FIELDS = ["dims", "family", "grid", "k_cols", "fields", "n_modes", "engine", "device_us",
          "mxu_us", "vpu_us", "copy_us", "dma_us", "wall_us", "rel_err", "status"]
BUCKETS = ["MXU matmul", "VPU / fusion", "reshape / copy", "DMA / HBM"]

GRIDS = [2 ** p for p in range(4, 13)]
BAND = [("reim", rfft_low_modes),
        ("coupled", rfft_coupled_low_modes),
        ("reduced", rfft_reduced_low_modes)]
STAGED = [("reim", lambda x: rfft_half(x)),
          ("coupled", lambda x: rfft_coupled_reim(x, 0.5)),
          ("reduced", lambda x: rfft_reduced_reim(x, 0.5))]
FFT2 = FFT(dtype=F)


def modes_grid(grid, points=POINTS):
    """About `points` values of num_modes from 2 to N, even and log-spaced.

    Log rather than linear because every cost here is a ratio: the interesting structure sits
    near the narrow end, where an FNO lives, and a linear grid spends its points out on the flat.
    """
    raw = np.unique(np.round(np.geomspace(2, grid, points) / 2).astype(int) * 2)
    return [int(m) for m in raw if 2 <= m <= grid]


def width_for(grid):
    """Signals per 1-D cell, holding the element count fixed so every grid is compute-bound.

    A width fixed across grids measures the launch and setup path at the small end, where the
    matmul is too short to fill the unit. Holding the element count instead puts every grid on
    the same side of that floor; the cap keeps the smallest grids from going memory-bound.
    """
    return int(min(K_CAP, max(256, ELEMENTS // grid)))


def fields_for(grid):
    """Fields per 2-D cell, on the same element budget and capped at a sane batch."""
    return int(min(256, max(1, ELEMENTS // (grid * grid))))


def row_pass(engine, x, n_modes):
    """(n, c, h, w) real -> (re, im) each (n, c, n_modes, w), the engine doing the row axis."""
    n, c, h, w = x.shape
    flat = jnp.transpose(x, (2, 0, 1, 3)).reshape(h, n * c * w)
    re, im = engine(flat, n_modes)
    rows = re.shape[0]
    return (jnp.transpose(re.reshape(rows, n, c, w), (1, 2, 0, 3)),
            jnp.transpose(im.reshape(rows, n, c, w), (1, 2, 0, 3)))


def two_d(engine, x, n_modes):
    """The whole separable transform: engine on the real rows, complex pass on the columns."""
    re, im = row_pass(engine, x, n_modes)
    return FFT2.cols_forward(re, im, FFT2.half_band(n_modes), x.shape[-1])


def native_1d(x, n_modes):
    return jnp.fft.rfft(x, axis=0)[:n_modes]


def native_2d(x, n_modes):
    return jnp.fft.rfft2(x, axes=(-2, -1))[:, :, :n_modes, :max(1, n_modes // 2)]


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


def err_1d(engine, host, n_modes):
    re_, im_ = engine(jnp.asarray(host), n_modes)
    got = np.asarray(re_, np.float64) + 1j * np.asarray(im_, np.float64)
    exact = np.fft.fft(host.astype(np.float64), axis=0)[:n_modes]
    return float(np.abs(got - exact).max() / np.abs(exact).max())


def err_2d(engine, host, n_modes):
    re_, im_ = two_d(engine, jnp.asarray(host), n_modes)
    got = np.asarray(re_, np.float64) + 1j * np.asarray(im_, np.float64)
    full = np.fft.fft2(host.astype(np.float64), axes=(-2, -1))
    cols = [int(c) for c in FFT2.half_band(n_modes)]
    exact = full[:, :, :n_modes][:, :, :, cols]
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
        return f"{row['engine']} {float(row['device_us']):8.2f}"
    except Exception as exc:
        row["status"] = type(exc).__name__
        return f"{row['engine']}:{row['status'][:14]}"


def flush(rows_out):
    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)


def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    rng = np.random.default_rng(0)
    rows_out = []
    blank = {f: "" for f in FIELDS}

    for grid in GRIDS:
        modes = modes_grid(grid)
        fields = fields_for(grid)
        width = width_for(grid)
        host1 = rng.normal(size=(grid, width)).astype(np.float32)
        host2 = rng.normal(size=(fields, 1, grid, grid)).astype(np.float32)
        x1, x2 = jnp.asarray(host1), jnp.asarray(host2)
        check1, check2 = host1[:, :32], host2[:CHECK_FIELDS]
        print(f"\nN={grid}  1-D K={width}  2-D fields={fields}  "
              f"{len(modes)} values of num_modes", flush=True)

        for n_modes in modes:
            for dims in (1, 2):
                line = []
                for name, engine in BAND:
                    if dims == 1:
                        fn = jax.jit(lambda z, _e=engine, _m=n_modes: _e(z, _m))
                        chk = lambda _e=engine, _m=n_modes: err_1d(_e, check1, _m)
                        data = x1
                    else:
                        fn = jax.jit(lambda z, _e=engine, _m=n_modes: two_d(_e, z, _m))
                        chk = lambda _e=engine, _m=n_modes: err_2d(_e, check2, _m)
                        data = x2
                    row = dict(blank, dims=dims, family="band", grid=grid,
                               k_cols=width if dims == 1 else "",
                               fields="" if dims == 1 else fields,
                               n_modes=n_modes, engine=name)
                    line.append(fill(row, fn, data,
                                     f"grid_{dims}d_{grid}_{n_modes}_{name}", chk))
                    rows_out.append(row)

                base = native_1d if dims == 1 else native_2d
                fn = jax.jit(lambda z, _m=n_modes, _f=base: _f(z, _m))
                row = dict(blank, dims=dims, family="band", grid=grid,
                           k_cols=width if dims == 1 else "",
                           fields="" if dims == 1 else fields,
                           n_modes=n_modes, engine="native")
                line.append(fill(row, fn, x1 if dims == 1 else x2,
                                 f"grid_{dims}d_{grid}_{n_modes}_native"))
                rows_out.append(row)
                print(f"  {dims}-D m={n_modes:<5} " + "  ".join(line), flush=True)

        line = []
        for name, engine in STAGED:
            fn = jax.jit(engine)
            row = dict(blank, dims=1, family="staged", grid=grid, k_cols=width, fields="",
                       n_modes=grid // 2, engine=name)
            line.append(fill(row, fn, x1, f"grid_staged_{grid}_{name}",
                             lambda _e=engine: err_1d(lambda z, _m: _e(z),
                                                      check1, grid // 2)))
            rows_out.append(row)
        print("  1-D staged   " + "  ".join(line), flush=True)
        flush(rows_out)

    flush(rows_out)
    bad = [r for r in rows_out if r["rel_err"]]
    if bad:
        worst = max(bad, key=lambda r: float(r["rel_err"]))
        print(f"\nworst rel_err {worst['rel_err']} at {worst['dims']}-D N={worst['grid']} "
              f"m={worst['n_modes']} {worst['engine']}", flush=True)
    failed = [r for r in rows_out if not r["status"].startswith("ok")]
    print(f"radix {RADIX}, wrote {OUT}  ({len(rows_out)} rows, {len(failed)} failed)",
          flush=True)
    for r in failed[:12]:
        print(f"   {r['dims']}-D N={r['grid']} m={r['n_modes']} {r['engine']}: {r['status']}",
              flush=True)
    print("GRID_DONE", flush=True)


if __name__ == "__main__":
    main()
