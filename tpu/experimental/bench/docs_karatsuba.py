"""
Is the Karatsuba complex product actually worth it on a v5e?

The identity trades one matmul for extra elementwise work:

  naive      4 matmuls:  re = C_cos @ x_re - C_sin @ x_im ,  im = C_cos @ x_im + C_sin @ x_re
  karatsuba  3 matmuls:  a = C_cos @ x_re, b = C_sin @ x_im, c = (C_cos + C_sin) @ (x_re + x_im)
                         re = a - b,  im = c - a - b

That is 25% fewer matmuls, but it needs a full elementwise pass to form (x_re + x_im) -- reading
both operands and writing a third array -- plus two extra subtractions on the output. On a kernel
whose runtime is set by bytes rather than arithmetic, saving a matmul while ADDING a memory pass
is not obviously a win, which is what this measures.

Reported at three levels: the bare complex product, a full staged complex FFT built on each, and
the 2-corner spectral conv.

    PYTHONPATH=~/nop ~/nopenv/bin/python -m tpu.experimental.bench.docs_karatsuba
"""
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import ACC, F, RADIX, dft_matrix_reim, leaf_matmul, stage_matmul

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_kara")
REPS = 20


def complex_matmul_naive(matmul, cos, sin, cos_plus_sin, x_re, x_im):
    """The straightforward four-matmul complex product; cos_plus_sin is unused."""
    return (matmul(cos, x_re) - matmul(sin, x_im),
            matmul(cos, x_im) + matmul(sin, x_re))


def complex_matmul_karatsuba(matmul, cos, sin, cos_plus_sin, x_re, x_im):
    a = matmul(cos, x_re)
    b = matmul(sin, x_im)
    c = matmul(cos_plus_sin, x_re + x_im)
    return a - b, c - a - b


def fft_staged_with(product, x_re, x_im, n, dtype=F):
    """fft_staged_complex, but with the complex product swapped for `product`."""
    cur_re, cur_im, length, batch, stages = x_re, x_im, n, x_re.shape[1], []
    while length > RADIX and length % RADIX == 0:
        n_cols = length // RADIX
        cos, sin, cps = dft_matrix_reim(RADIX, dtype)
        s_re, s_im = product(stage_matmul, cos, sin, cps,
                             cur_re.reshape(RADIX, n_cols, batch),
                             cur_im.reshape(RADIX, n_cols, batch))
        rows, cols = jnp.arange(RADIX)[:, None], jnp.arange(n_cols)[None, :]
        ang = (-2 * jnp.pi * rows * cols / length).astype(F)
        tc, tsn = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
        cur_re = (s_re * tc - s_im * tsn).transpose(1, 0, 2).reshape(n_cols, RADIX * batch)
        cur_im = (s_re * tsn + s_im * tc).transpose(1, 0, 2).reshape(n_cols, RADIX * batch)
        stages.append(RADIX)
        length, batch = n_cols, RADIX * batch
    cos, sin, cps = dft_matrix_reim(length, dtype)
    out_re, out_im = product(leaf_matmul, cos, sin, cps, cur_re, cur_im)
    ln = length
    for r in reversed(stages):
        out_re = out_re.reshape(ln, r, -1).reshape(ln * r, -1)
        out_im = out_im.reshape(ln, r, -1).reshape(ln * r, -1)
        ln *= r
    return out_re, out_im


def measure(tag, fn, *args):
    run = jax.jit(fn)
    jax.block_until_ready(run(*args))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter(); jax.block_until_ready(run(*args)); ts.append(time.perf_counter() - t0)
    wall = sorted(ts)[len(ts) // 2] * 1e6
    try:
        c = jax.jit(fn).lower(*args).compile().cost_analysis()
        if isinstance(c, list):
            c = c[0]
        return wall, float(c.get("flops", 0.0)), float(c.get("bytes accessed", 0.0))
    except Exception:
        return wall, float("nan"), float("nan")


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("matmul precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []

    print("\n### bare complex product, (n x n) matrix against (n x K)", flush=True)
    print(f"  {'n':<7}{'K':<8}{'variant':<12}{'wall us':>10}{'GFLOP':>9}{'HBM MB':>10}{'rel diff':>11}", flush=True)
    for n, K in ((128, 8192), (128, 32768), (512, 8192)):
        cos, sin, cps = dft_matrix_reim(n, F)
        xr = jnp.asarray(rng.standard_normal((n, K)).astype(np.float32))
        xi = jnp.asarray(rng.standard_normal((n, K)).astype(np.float32))
        ref = None
        for name, prod in (("naive", complex_matmul_naive), ("karatsuba", complex_matmul_karatsuba)):
            fn = lambda a, b, p=prod: p(leaf_matmul, cos, sin, cps, a, b)
            wall, fl, hbm = measure(f"prod_{name}_n{n}_K{K}", fn, xr, xi)
            out = jax.jit(fn)(xr, xi)
            got = np.asarray(out[0], np.float64) + 1j * np.asarray(out[1], np.float64)
            if ref is None:
                ref, diff = got, 0.0
            else:
                diff = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
            print(f"  {n:<7}{K:<8}{name:<12}{wall:>10.1f}{fl/1e9:>9.3f}{hbm/1e6:>10.1f}{diff:>11.1e}", flush=True)
            rows.append({"level": "product", "n": n, "k": K, "variant": name, "wall_us": wall,
                         "gflop": fl / 1e9, "hbm_mb": hbm / 1e6, "rel_diff": diff})

    print("\n### full staged complex FFT (K=256)", flush=True)
    print(f"  {'N':<9}{'variant':<12}{'wall us':>10}{'GFLOP':>9}{'HBM MB':>10}{'rel err':>11}", flush=True)
    for n in (1024, 4096, 16384):
        xr = jnp.asarray(rng.standard_normal((n, 256)).astype(np.float32))
        xi = jnp.asarray(rng.standard_normal((n, 256)).astype(np.float32))
        truth = np.fft.fft(np.asarray(xr, np.float64) + 1j * np.asarray(xi, np.float64), axis=0)
        for name, prod in (("naive", complex_matmul_naive), ("karatsuba", complex_matmul_karatsuba)):
            fn = lambda a, b, p=prod: fft_staged_with(p, a, b, n)
            wall, fl, hbm = measure(f"fft_{name}_N{n}", fn, xr, xi)
            out = jax.jit(fn)(xr, xi)
            got = np.asarray(out[0], np.float64) + 1j * np.asarray(out[1], np.float64)
            err = float(np.max(np.abs(got - truth)) / np.max(np.abs(truth)))
            print(f"  {n:<9}{name:<12}{wall:>10.1f}{fl/1e9:>9.3f}{hbm/1e6:>10.1f}{err:>11.1e}", flush=True)
            rows.append({"level": "fft", "n": n, "k": 256, "variant": name, "wall_us": wall,
                         "gflop": fl / 1e9, "hbm_mb": hbm / 1e6, "rel_diff": err})

    with open(os.environ.get("DOCS_KAR_CSV", "/tmp/docs_karatsuba.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader(); w.writerows(rows)
    print("\nKARATSUBA_DONE", flush=True)


if __name__ == "__main__":
    main()
