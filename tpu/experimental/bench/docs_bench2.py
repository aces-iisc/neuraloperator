"""
The three comparisons for the write-up. Every matmul runs at precision HIGHEST (fft/core.py sets
it globally on import; main() asserts it), and every configuration also writes an xprof trace so
device time can be recovered off-box -- wall-clock alone has a ~180us dispatch floor that swamps
the small-N rows.

  part 1  1-D engines swept over N, f32 and bf16, timed AND checked for accuracy.
  part 2  FNO inference: the mode band set wide (n_modes = grid/2) and narrow, f32 and bf16.
  part 3  the coupled packing used to produce a narrow band directly, against the partial DFT.

    PYTHONPATH=~/nop ~/nopenv/bin/python -m tpu.experimental.bench.docs_bench2
"""
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import BF, F  # noqa: F401  (import sets precision HIGHEST globally)
from ...fft.rfft1d import rfft_half, rfft_low_modes
from ...fno.layers import _init_spectral_weight
from ...fno.spectral import spectral_conv
from ..fft.tricks import rfft_coupled_low_modes, rfft_coupled_reim, rfft_reduced_reim
from ..fno import reference
from ..fno.spectral_variants import spectral_2corner_herm, spectral_2corner_hybrid

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof2")
REPS = 20
ROWS = []


def _trace_and_time(tag, fn, *args):
    """Run under a profiler trace (compile/warmup outside it) and also take a wall-clock median."""
    import time
    run = jax.jit(fn)
    jax.block_until_ready(run(*args))
    target = os.path.join(LOGDIR, tag)
    ts = []
    with jax.profiler.trace(target):
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(run(*args))
            ts.append(time.perf_counter() - t0)
    wall = sorted(ts)[len(ts) // 2] * 1e6
    try:
        cost = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(cost, list):
            cost = cost[0]
        hbm = float(cost.get("bytes accessed", 0.0))
        flops = float(cost.get("flops", 0.0))
    except Exception:
        hbm = flops = float("nan")
    return wall, hbm, flops, target


def record(part, case, engine, dtype, fn, args, ref=None, extra=None):
    tag = f"{part}_{case}_{engine}_{dtype}".replace("/", "-").replace(" ", "")
    wall, hbm, flops, target = _trace_and_time(tag, fn, *args)
    out = jax.jit(fn)(*args)
    err = float("nan")
    if ref is not None:
        got = np.asarray(out[0], np.float64) + 1j * np.asarray(out[1], np.float64) \
            if isinstance(out, tuple) else np.asarray(out, np.float64)
        err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-30))
    row = {"part": part, "case": case, "engine": engine, "dtype": dtype,
           "wall_us": wall, "hbm_mb": hbm / 1e6, "gflop": flops / 1e9,
           "rel_err": err, "trace": os.path.relpath(target, LOGDIR)}
    row.update(extra or {})
    ROWS.append(row)
    print(f"  {case:<16}{engine:<20}{dtype:<6}{wall:>9.1f}us{hbm/1e6:>10.2f}MB"
          f"{err:>12.2e}", flush=True)
    return row


def part1_fft_1d():
    print("\n=== part 1: 1-D transforms (K=256) ===", flush=True)
    print(f"  {'case':<16}{'engine':<20}{'dt':<6}{'wall':>11}{'HBM':>12}{'rel err':>12}", flush=True)
    K = 256
    for n in (128, 256, 512, 1024, 4096, 16384):
        x32 = jnp.asarray(np.random.randn(n, K).astype(np.float32))
        x16 = x32.astype(BF)
        exact = np.fft.fft(np.asarray(x32, np.float64), axis=0)   # float64 ground truth
        half = exact[: n // 2]
        case = f"N={n}"
        record("fft1d", case, "reim", "f32", lambda z: rfft_half(z, F), (x32,), half)
        record("fft1d", case, "reim", "bf16", lambda z: rfft_half(z, BF), (x16,), half)
        record("fft1d", case, "coupled", "f32", lambda z: rfft_coupled_reim(z, 0.5, F), (x32,), half)
        record("fft1d", case, "coupled", "bf16", lambda z: rfft_coupled_reim(z, 0.5, BF), (x16,), half)
        record("fft1d", case, "reduced", "f32", lambda z: rfft_reduced_reim(z, 0.5, F), (x32,), half)
        # jnp.rfft returns n//2+1 bins; trim to n//2 so every engine is scored on the same band
        record("fft1d", case, "jnp.rfft", "c64",
               lambda z, h=n // 2: (jnp.fft.rfft(z, axis=0).real[:h],
                                    jnp.fft.rfft(z, axis=0).imag[:h]),
               (x32,), half)
        m = min(16, n // 2)
        record("fft1d", case, f"partial(m={m})", "f32",
               lambda z: rfft_low_modes(z, m, F), (x32,), exact[:m], extra={"n_modes": m})


def part2_fno_inference():
    print("\n=== part 2: FNO inference ===", flush=True)
    print(f"  {'case':<16}{'engine':<20}{'dt':<6}{'wall':>11}{'HBM':>12}{'rel err':>12}", flush=True)
    configs = [(128, 24, 8), (256, 16, 4), (1024, 8, 1)]
    for grid, channels, batch in configs:
        for m in (grid // 2, 12):
            key = jax.random.PRNGKey(0)
            w = _init_spectral_weight(key, channels, channels, m)
            x = jnp.asarray(np.random.randn(batch, channels, grid, grid).astype(np.float32))
            case = f"{grid}^2 m={m}"
            base = np.asarray(jax.jit(lambda w_, z: spectral_conv(w_, z, m))(w, x), np.float64)

            record("fno", case, "reim+partial", "f32",
                   lambda w_, z: spectral_conv(w_, z, m), (w, x), base,
                   extra={"grid": grid, "n_modes": m, "channels": channels})
            record("fno", case, "reim+partial", "bf16",
                   lambda w_, z: spectral_conv(w_, z, m, BF), (w, x), base,
                   extra={"grid": grid, "n_modes": m, "channels": channels})
            for name, fn in [("reim full+select", spectral_2corner_hybrid),
                             ("reim hermitian", spectral_2corner_herm),
                             ("jnp.fft", reference.spectral_conv)]:
                try:
                    record("fno", case, name, "f32",
                           lambda w_, z, f=fn: f(w_, z, m), (w, x), base,
                           extra={"grid": grid, "n_modes": m, "channels": channels})
                except Exception as exc:
                    print(f"    skip {name} {case}: {type(exc).__name__}", flush=True)


def part3_coupled_low_modes():
    print("\n=== part 3: coupled packing for a narrow band (n_modes=12) ===", flush=True)
    print(f"  {'case':<16}{'engine':<20}{'dt':<6}{'wall':>11}{'HBM':>12}{'rel err':>12}", flush=True)
    K, m = 256, 12
    for n in (128, 256, 512, 1024, 4096, 16384):
        x = jnp.asarray(np.random.randn(n, K).astype(np.float32))
        exact = np.fft.fft(np.asarray(x, np.float64), axis=0)[:m]
        case = f"N={n}"
        record("coupled_low", case, "partial", "f32",
               lambda z: rfft_low_modes(z, m, F), (x,), exact, extra={"n_modes": m})
        record("coupled_low", case, "coupled-partial", "f32",
               lambda z: rfft_coupled_low_modes(z, m, F), (x,), exact, extra={"n_modes": m})
        record("coupled_low", case, "reim+truncate", "f32",
               lambda z: tuple(a[:m] for a in rfft_half(z, F)), (x,), exact, extra={"n_modes": m})


def main():
    prec = jax.config.jax_default_matmul_precision
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"matmul precision = {prec}", flush=True)
    assert str(prec).lower().endswith("highest"), f"expected HIGHEST, got {prec}"
    os.makedirs(LOGDIR, exist_ok=True)

    want = os.environ.get("DOCS_PARTS", "1,2,3").split(",")
    if "1" in want:
        part1_fft_1d()
    if "3" in want:
        part3_coupled_low_modes()
    if "2" in want:
        part2_fno_inference()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.environ.get("DOCS_B2_CSV", "/tmp/docs_bench2.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(ROWS)
    print(f"\nwrote /tmp/docs_bench2.csv ({len(ROWS)} rows)", flush=True)
    print("DOCS_BENCH2_DONE", flush=True)


if __name__ == "__main__":
    main()
