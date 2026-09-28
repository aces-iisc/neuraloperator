"""Engine sweep N=32..4096 plus a K_TILE sweep, timed from traces. No global VMEM flag."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fft.rfft1d import rfft_direct, rfft_half, rfft_low_modes
from ..fft.pallas1d import (pallas_direct, pallas_half, pallas_partial,
                            pallas_partial_cached, pallas_reduced)
from ..fft.tricks import rfft_reduced_reim

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_p2")
CSV = os.environ.get("DOCS_P2_CSV", "/tmp/docs_pallas2.csv")
FIELDS = ["part", "case", "engine", "k_tile", "grid", "wall_us", "rel_err", "status", "trace"]
REPS = 20
SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096)
K, MODES = 256, 16
K_BIG = 2048                      # part B: big enough that K_TILE actually splits the grid


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def run(tag, fn, *args):
    j = jax.jit(fn)
    jax.block_until_ready(j(*args))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(*args)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2]*1e6


def bench(rows, part, case, name, fn, args, truth, k_tile=0, grid=0):
    tag = f"{part}_{case}_{name}_{k_tile}".replace(" ", "").replace("=", "")
    rec = {"part": part, "case": case, "engine": name, "k_tile": k_tile, "grid": grid}
    try:
        wall = run(tag, fn, *args)
        o = jax.jit(fn)(*args)
        g = np.asarray(o[0], np.float64) + 1j*np.asarray(o[1], np.float64)
        err = float(np.max(np.abs(g - truth))/np.max(np.abs(truth)))
        print(f"  {case:<9}{name:<20}{k_tile:>6}{wall:>9.1f}{err:>10.1e}", flush=True)
        rec.update(wall_us=wall, rel_err=err, status="ok", trace=tag)
    except Exception as exc:
        why = type(exc).__name__
        m = str(exc).lower()
        if "resource" in m or "memory" in m or "vmem" in m:
            why = "OOM"
        print(f"  {case:<9}{name:<20}{k_tile:>6}{why:>9}   {str(exc)[:42]}", flush=True)
        rec.update(wall_us=float("nan"), rel_err=float("nan"), status=why, trace="")
    rows.append(rec); save(rows)


def part_engines(rows):
    print(f"\n=== A. engines, K={K}, one grid step ===", flush=True)
    print(f"  {'N':<9}{'engine':<20}{'K_TILE':>6}{'wall us':>9}{'rel err':>10}", flush=True)
    for n in SIZES:
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        t = np.fft.fft(np.asarray(x, np.float64), axis=0)
        half, low = t[:n//2], t[:min(MODES, n)]
        m = min(MODES, n)
        c = f"N={n}"
        bench(rows, "A", c, "xla reim",       lambda z: rfft_half(z, F), (x,), half, K, 1)
        bench(rows, "A", c, "pallas reim",    lambda z: pallas_half(z, K, False, F), (x,), half, K, 1)
        bench(rows, "A", c, "xla reduced",    lambda z: rfft_reduced_reim(z, 0.5, F), (x,), half, K, 1)
        bench(rows, "A", c, "pallas reduced", lambda z: pallas_reduced(z, K, False, F), (x,), half, K, 1)
        bench(rows, "A", c, "xla direct",     lambda z: rfft_direct(z, True, F), (x,), half, K, 1)
        bench(rows, "A", c, "pallas direct",  lambda z: pallas_direct(z, K, False, F), (x,), half, K, 1)
        bench(rows, "A", c, "xla partial",    lambda z: rfft_low_modes(z, m, F), (x,), low, K, 1)
        bench(rows, "A", c, "pallas partial", lambda z: pallas_partial(z, m, K, False, F), (x,), low, K, 1)
        bench(rows, "A", c, "pallas partial cached",
              lambda z: pallas_partial_cached(z, m, K, False, F), (x,), low, K, 1)
        print(flush=True)


def part_ktile(rows):
    print(f"\n=== B. K_TILE sweep, K={K_BIG} (grid = K/K_TILE) ===", flush=True)
    print(f"  {'N':<9}{'engine':<20}{'K_TILE':>6}{'wall us':>9}{'rel err':>10}", flush=True)
    for n in (128, 512, 2048):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K_BIG)).astype(np.float32))
        t = np.fft.fft(np.asarray(x, np.float64), axis=0)
        half, low = t[:n//2], t[:MODES]
        c = f"N={n}"
        bench(rows, "B", c, "xla partial", lambda z: rfft_low_modes(z, MODES, F), (x,), low, K_BIG, 1)
        bench(rows, "B", c, "xla reim", lambda z: rfft_half(z, F), (x,), half, K_BIG, 1)
        for kt in (128, 256, 512, 1024, 2048):
            g = K_BIG // kt
            bench(rows, "B", c, "pallas partial",
                  lambda z, t=kt: pallas_partial(z, MODES, t, False, F), (x,), low, kt, g)
            bench(rows, "B", c, "pallas partial cached",
                  lambda z, t=kt: pallas_partial_cached(z, MODES, t, False, F), (x,), low, kt, g)
            bench(rows, "B", c, "pallas reim",
                  lambda z, t=kt: pallas_half(z, t, False, F), (x,), half, kt, g)
        print(flush=True)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    part_engines(rows)
    part_ktile(rows)
    save(rows)
    print("P2_DONE", flush=True)


if __name__ == "__main__":
    main()
