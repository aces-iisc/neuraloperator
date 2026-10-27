"""The three real-input transforms over a (B, N, num_modes) volume, on one v5e.

The companion of the GPU sweep of the same name and the same experiment. The grid sweep held
the element count fixed so that every grid was given enough work to fill the machine; this one
holds the BATCH fixed and sweeps it, which is the question a caller actually faces -- the batch
is given, not chosen.

B is the batch in both dimensionalities: independent signals in 1-D, fields in 2-D. The data is
therefore B*N floats in 1-D and B*N^2 in 2-D, so 1-D covers the whole box while 2-D is a wedge.
One 4096^2 field is 67 MiB on its own, and the row pass holds a field set and its permuted copy
at the same time. Cells past the budget are recorded as skipped rather than dropped.

Time is DEVICE time from this run's own traces, because at small B the kernel is a couple of
microseconds against a dispatch floor two orders of magnitude larger -- and separating those two
is most of the point of sweeping B at all. Each trace is parsed and then deleted. Every cell is
checked against a float64 NumPy transform before it is timed.
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

OUT = os.path.expanduser(os.environ.get("BATCH_OUT", "~/repro_out/engines_batch_tpu.csv"))
TRACES = os.path.expanduser(os.environ.get("BATCH_TRACES", "~/batch_traces"))
REPS = int(os.environ.get("BATCH_REPS", "6"))
POINTS = int(os.environ.get("BATCH_POINTS", "8"))
BUDGET = float(os.environ.get("BATCH_BUDGET", "3.0")) * 2 ** 30
CHECK_COLS = 32
CHECK_FIELDS = 2
FIELDS = ["dims", "family", "grid", "B", "n_modes", "engine", "device_us",
          "mxu_us", "vpu_us", "copy_us", "dma_us", "wall_us", "rel_err", "status"]
BUCKETS = ["MXU matmul", "VPU / fusion", "reshape / copy", "DMA / HBM"]

GRIDS = [2 ** p for p in range(4, 13)]
BATCHES = [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
BAND = [("reim", rfft_low_modes),
        ("coupled", rfft_coupled_low_modes),
        ("reduced", rfft_reduced_low_modes)]
STAGED = [("reim", lambda x: rfft_half(x)),
          ("coupled", lambda x: rfft_coupled_reim(x, 0.5)),
          ("reduced", lambda x: rfft_reduced_reim(x, 0.5))]
FFT2 = FFT(dtype=F)


def modes_grid(grid, points=POINTS):
    """About `points` values of num_modes from 2 to N, even and log-spaced."""
    raw = np.unique(np.round(np.geomspace(2, grid, points) / 2).astype(int) * 2)
    return [int(m) for m in raw if 2 <= m <= grid]


def input_bytes(dims, grid, batch):
    return batch * grid * 4 if dims == 1 else batch * grid * grid * 4


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
        row["status"] = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
        return f"{row['engine']}:{row['status'][:12]}"


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
                if dims == 1:
                    host = rng.normal(size=(grid, batch)).astype(np.float32)
                    check = host[:, :min(CHECK_COLS, batch)]
                else:
                    host = rng.normal(size=(batch, 1, grid, grid)).astype(np.float32)
                    check = host[:min(CHECK_FIELDS, batch)]
                x = jnp.asarray(host)
                del host
                print(f"\n  {dims}-D N={grid}  B={batch}  "
                      f"{want / 2 ** 20:.1f} MiB  {len(modes)} bands", flush=True)
                for n_modes in modes:
                    line = []
                    for name, engine in BAND:
                        if dims == 1:
                            fn = jax.jit(lambda z, _e=engine, _m=n_modes: _e(z, _m))
                            chk = lambda _e=engine, _m=n_modes: err_1d(_e, check, _m)
                        else:
                            fn = jax.jit(lambda z, _e=engine, _m=n_modes: two_d(_e, z, _m))
                            chk = lambda _e=engine, _m=n_modes: err_2d(_e, check, _m)
                        row = dict(blank, dims=dims, family="band", grid=grid, B=batch,
                                   n_modes=n_modes, engine=name)
                        line.append(fill(row, fn, x,
                                         f"b{dims}_{grid}_{batch}_{n_modes}_{name}", chk))
                        rows_out.append(row)
                    base = native_1d if dims == 1 else native_2d
                    fn = jax.jit(lambda z, _m=n_modes, _f=base: _f(z, _m))
                    row = dict(blank, dims=dims, family="band", grid=grid, B=batch,
                               n_modes=n_modes, engine="native")
                    line.append(fill(row, fn, x,
                                     f"b{dims}_{grid}_{batch}_{n_modes}_native"))
                    rows_out.append(row)
                    print(f"    m={n_modes:<5} " + "  ".join(line), flush=True)
                if dims == 1:
                    line = []
                    for name, engine in STAGED:
                        row = dict(blank, dims=1, family="staged", grid=grid, B=batch,
                                   n_modes=grid // 2, engine=name)
                        line.append(fill(row, jax.jit(engine), x,
                                         f"bstaged_{grid}_{batch}_{name}",
                                         lambda _e=engine: err_1d(lambda z, _m: _e(z),
                                                                  check, grid // 2)))
                        rows_out.append(row)
                    print("    staged  " + "  ".join(line), flush=True)
                del x
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
