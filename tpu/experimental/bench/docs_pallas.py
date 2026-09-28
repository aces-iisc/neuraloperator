"""Pallas engines against their XLA counterparts. OOM / lowering failures are reported, not hidden."""
import csv, os, time
import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fft.rfft1d import rfft_direct, rfft_half, rfft_low_modes
from ...fft.rfft2d import rfft2_full, rfft2_low_modes
from ..fft.pallas1d import (pallas_coupled, pallas_direct, pallas_half, pallas_partial,
                            pallas_reduced)
from ..fft.pallas2d import pallas_rfft2
from ..fft.tricks import rfft_coupled_reim, rfft_reduced_reim

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_pallas")
CSV = os.environ.get("DOCS_PALLAS_CSV", "/tmp/docs_pallas.csv")
FIELDS = ["part", "case", "engine", "wall_us", "hbm_mb", "rel_err", "status", "trace"]
REPS = 20
K, MODES = 256, 16
PALLAS_2D_MAX = int(os.environ.get("PALLAS_2D_MAX", "128"))


def run(tag, fn, *args):
    j = jax.jit(fn)
    jax.block_until_ready(j(*args))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(j(*args)); ts.append(time.perf_counter()-t0)
    wall = sorted(ts)[len(ts)//2]*1e6
    try:
        c = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(c, list): c = c[0]
        hbm = float(c.get("bytes accessed", 0.0))
    except Exception:
        hbm = float("nan")
    return wall, hbm


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def bench(rows, part, case, name, fn, args, truth=None):
    tag = f"{part}_{case}_{name}".replace(" ", "").replace("=", "")
    try:
        wall, hbm = run(tag, fn, *args)
        err = float("nan")
        if truth is not None:
            o = jax.jit(fn)(*args)
            g = np.asarray(o[0], np.float64) + 1j*np.asarray(o[1], np.float64)
            err = float(np.max(np.abs(g - truth))/np.max(np.abs(truth)))
        print(f"  {case:<12}{name:<18}{wall:>10.1f}{hbm/1e6:>10.2f}{err:>11.1e}", flush=True)
        rows.append({"part": part, "case": case, "engine": name, "wall_us": wall,
                     "hbm_mb": hbm/1e6, "rel_err": err, "status": "ok", "trace": tag})
        save(rows)
    except Exception as exc:
        why = type(exc).__name__
        msg = str(exc).lower()
        if "resource" in msg or "memory" in msg or "vmem" in msg or "oom" in msg:
            why = "OOM"
        print(f"  {case:<12}{name:<18}{why:>10}   {str(exc)[:44]}", flush=True)
        rows.append({"part": part, "case": case, "engine": name, "wall_us": float("nan"),
                     "hbm_mb": float("nan"), "rel_err": float("nan"), "status": why, "trace": ""})
        save(rows)


def part1(rows):
    print("\n=== 1-D: Pallas (VMEM-resident) against XLA, K=256 ===", flush=True)
    print(f"  {'N':<12}{'engine':<18}{'wall us':>10}{'HBM MB':>10}{'rel err':>11}", flush=True)
    for n in (128, 256, 512, 1024, 2048, 4096):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        t = np.fft.fft(np.asarray(x, np.float64), axis=0)
        half, low = t[:n//2], t[:MODES]
        c = f"N={n}"
        bench(rows, "1d", c, "xla reim",     lambda z: rfft_half(z, F), (x,), half)
        bench(rows, "1d", c, "pallas reim",  lambda z: pallas_half(z, K, False, F), (x,), half)
        bench(rows, "1d", c, "xla coupled",  lambda z: rfft_coupled_reim(z, 0.5, F), (x,), half)
        bench(rows, "1d", c, "pallas coupled", lambda z: pallas_coupled(z, K, False, F), (x,), half)
        bench(rows, "1d", c, "xla reduced",  lambda z: rfft_reduced_reim(z, 0.5, F), (x,), half)
        bench(rows, "1d", c, "pallas reduced", lambda z: pallas_reduced(z, K, False, F), (x,), half)
        bench(rows, "1d", c, "xla direct",   lambda z: rfft_direct(z, True, F), (x,), half)
        bench(rows, "1d", c, "pallas direct", lambda z: pallas_direct(z, K, False, F), (x,), half)
        bench(rows, "1d", c, "xla partial",  lambda z: rfft_low_modes(z, MODES, F), (x,), low)
        bench(rows, "1d", c, "pallas partial", lambda z: pallas_partial(z, MODES, K, False, F), (x,), low)
        print(flush=True)


def part2(rows):
    print("\n=== 2-D: the transform an FNO layer runs, K=64 ===", flush=True)
    print(f"  {'grid':<12}{'engine':<18}{'wall us':>10}{'HBM MB':>10}{'rel err':>11}", flush=True)
    for g in (32, 64, 128, 256):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((g, g, 64)).astype(np.float32))
        t = np.fft.fft2(np.asarray(x, np.float64), axes=(0, 1))
        m = min(MODES, g//2)
        c = f"{g}x{g}"
        bench(rows, "2d", c, "xla reim",    lambda z: rfft2_full(z, F), (x,), t[:g//2])
        if g <= PALLAS_2D_MAX:
            bench(rows, "2d", c, "pallas reim", lambda z: pallas_rfft2(z, 64, False, F), (x,), t[:g//2])
        else:
            # 256x256x64 hangs the TPU hard (VM goes health=TIMEOUT, ssh dies) -- reproduced twice.
            print(f"  {c:<12}{'pallas reim':<18}{'HANG':>10}   skipped: wedges the device", flush=True)
            rows.append({"part": "2d", "case": c, "engine": "pallas reim", "wall_us": float("nan"),
                         "hbm_mb": float("nan"), "rel_err": float("nan"), "status": "HANG",
                         "trace": ""})
            save(rows)
        bench(rows, "2d", c, "xla partial", lambda z: rfft2_low_modes(z, m/g, F), (x,), t[:m, :m])
        print(flush=True)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    part1(rows)
    part2(rows)
    save(rows)
    print("PALLAS_DONE", flush=True)


if __name__ == "__main__":
    main()
