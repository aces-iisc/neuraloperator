"""
The +/- band on the row axis two ways, at n_modes = 12.

The two corners of the band are conjugates of each other for a real field, so the negative half
can either be COMPUTED (evaluate all 2m rows of the DFT matrix) or MIRRORED (evaluate m+1 rows and
build the rest by reversing and conjugating). Mirroring does half the arithmetic; reversal is a
memory operation. On a memory-bound chip that is not obviously the better trade, which is the point
of measuring it.

    PYTHONPATH=~/nop ~/nopenv/bin/python -m tpu.experimental.bench.docs_hermitian
"""
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import ACC, F  # noqa: F401  (sets precision HIGHEST globally)

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_herm")
REPS = 20


def _dft_rows(row_idx, length, dtype=F):
    k = jnp.asarray(row_idx, F)[:, None]
    n = jnp.arange(length, dtype=F)[None, :]
    angle = -2.0 * jnp.pi * k * n / length
    return jnp.cos(angle).astype(dtype), jnp.sin(angle).astype(dtype)


def band_computed(x, n_modes, dtype=F):
    """Evaluate every row of the band directly: k in [0,m) U [N-m,N), i.e. 2m rows."""
    n = x.shape[0]
    rows = jnp.concatenate([jnp.arange(n_modes), jnp.arange(n - n_modes, n)])
    cos, sin = _dft_rows(rows, n, dtype)
    z = x.astype(dtype)
    return (jnp.matmul(cos, z, preferred_element_type=ACC),
            jnp.matmul(sin, z, preferred_element_type=ACC))


def band_mirrored(x, n_modes, dtype=F):
    """Evaluate only k in [0,m], then build the negative half by reversing and conjugating,
    using X[N-k] = conj(X[k]).  m+1 rows of arithmetic plus a reverse."""
    n = x.shape[0]
    cos, sin = _dft_rows(jnp.arange(n_modes + 1), n, dtype)
    z = x.astype(dtype)
    re = jnp.matmul(cos, z, preferred_element_type=ACC)
    im = jnp.matmul(sin, z, preferred_element_type=ACC)
    neg_re = re[n_modes:0:-1]          # rows m..1 reversed  -> k = N-m .. N-1
    neg_im = -im[n_modes:0:-1]         # conjugate
    return (jnp.concatenate([re[:n_modes], neg_re], axis=0),
            jnp.concatenate([im[:n_modes], neg_im], axis=0))


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("matmul precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    K, m = 256, 12
    rows = []
    print(f"\n  {'N':<8}{'engine':<20}{'wall us':>10}{'HBM MB':>10}{'rel err':>11}", flush=True)
    for n in (128, 1024):
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        exact = np.fft.fft(np.asarray(x, np.float64), axis=0)
        want = np.concatenate([exact[:m], exact[n - m:]], axis=0)
        for name, fn in [("computed", band_computed), ("mirrored", band_mirrored)]:
            run = jax.jit(lambda z, f=fn: f(z, m))
            jax.block_until_ready(run(x))
            tag = f"N{n}_{name}"
            ts = []
            with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                for _ in range(REPS):
                    t0 = time.perf_counter(); jax.block_until_ready(run(x)); ts.append(time.perf_counter() - t0)
            wall = sorted(ts)[len(ts) // 2] * 1e6
            out = run(x)
            got = np.asarray(out[0], np.float64) + 1j * np.asarray(out[1], np.float64)
            err = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
            try:
                c = jax.jit(lambda z, f=fn: f(z, m)).lower(x).compile().cost_analysis()
                if isinstance(c, list):
                    c = c[0]
                hbm = float(c.get("bytes accessed", 0.0))
            except Exception:
                hbm = float("nan")
            print(f"  {n:<8}{name:<20}{wall:>10.1f}{hbm/1e6:>10.2f}{err:>11.1e}", flush=True)
            rows.append({"n": n, "engine": name, "n_modes": m, "wall_us": wall,
                         "hbm_mb": hbm / 1e6, "rel_err": err, "trace": tag})
    with open("/tmp/docs_hermitian.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print("\nHERMITIAN_DONE", flush=True)


if __name__ == "__main__":
    main()
