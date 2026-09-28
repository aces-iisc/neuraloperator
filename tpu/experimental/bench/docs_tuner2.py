"""Autotuner: shapes compiled in parallel, weights passed as arguments, timed on device."""
import csv, os, time
from concurrent.futures import ThreadPoolExecutor

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import layers, model
from ...fno.autotune import neighbours_add, neighbours_mul, predict_batch
from ...fno.spectral import _E, _dft_rows

CSV = os.environ.get("DOCS_T2_CSV", "/tmp/docs_tuner2.csv")
ENGINE_CSV = os.environ.get("DOCS_T2_ENGINE_CSV", "/tmp/docs_tuner2_engine.csv")
FIELDS = ["case", "dtype", "n_modes", "batch", "us_per_sample", "status"]
ENGINE_FIELDS = ["case", "grid", "n_modes", "engine", "us_per_call", "status"]
ENGINE_BATCH = int(os.environ.get("DOCS_ENGINE_BATCH", "32"))
REPS = 10
WORKERS = int(os.environ.get("DOCS_WORKERS", "8"))
MODE_STEPS = int(os.environ.get("DOCS_MODE_STEPS", "1"))   # +-1 step of 8
BATCH_STEPS = int(os.environ.get("DOCS_BATCH_STEPS", "2"))  # +-2 powers of two
SERIAL = os.environ.get("DOCS_SERIAL", "0") == "1"

CASES = [("Burgers", 24, (17, 16), 16), ("Darcy-32", 24, (32, 32), 16),
         ("Darcy-64", 24, (64, 64), 16), ("NS-128", 64, (128, 128), 24)]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def _rows_partial(x, n_modes, dtype):
    """The row pass, held fixed while the column pass varies."""
    n_rows = x.shape[-2]
    half = n_modes // 2
    band = np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)])
    cos, sin = (a.astype(dtype) for a in _dft_rows(band, n_rows, -1))
    both = _E("ah,nchw->ncaw", jnp.concatenate([cos, sin], axis=0), x.astype(dtype))
    return both[:, :, :band.shape[0]], both[:, :, band.shape[0]:]


def fwd_col_partial(x, n_modes, dtype):
    """Column pass over the kept modes only."""
    n_cols, half = x.shape[-1], n_modes // 2
    rows_re, rows_im = _rows_partial(x, n_modes, dtype)
    cos, sin = (a.astype(dtype) for a in _dft_rows(np.arange(half), n_cols, -1))
    col = jnp.concatenate([cos, sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col, z)
    from_re, from_im = over(rows_re), over(rows_im)
    return (from_re[..., :half] - from_im[..., half:],
            from_re[..., half:] + from_im[..., :half])


def fwd_col_full(x, n_modes, dtype):
    """Column pass over every frequency, sliced to the kept modes afterwards."""
    n_cols, half = x.shape[-1], n_modes // 2
    rows_re, rows_im = _rows_partial(x, n_modes, dtype)
    cos, sin = (a.astype(dtype) for a in _dft_rows(np.arange(n_cols), n_cols, -1))
    col = jnp.concatenate([cos, sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col, z)
    from_re, from_im = over(rows_re), over(rows_im)
    re_cos, re_sin = from_re[..., :n_cols][..., :half], from_re[..., n_cols:][..., :half]
    im_cos, im_sin = from_im[..., :n_cols][..., :half], from_im[..., n_cols:][..., :half]
    return re_cos - im_sin, re_sin + im_cos


COL_ENGINES = {"partial": fwd_col_partial, "full": fwd_col_full}


def pick_col_engine(label, C, H, W, m, rows):
    """Time one forward transform, no GEMM and no inverse, and keep the faster column pass."""
    x = jnp.asarray(np.random.default_rng(0).standard_normal(
        (ENGINE_BATCH, C, H, W)).astype(np.float32))
    best, t0 = (None, 1e18), time.perf_counter()
    for name, fn in COL_ENGINES.items():
        try:
            f = jax.jit(lambda z, _f=fn: _f(z, m, F))
            jax.block_until_ready(f(x))
            ts = []
            for _ in range(REPS):
                s0 = time.perf_counter(); jax.block_until_ready(f(x))
                ts.append(time.perf_counter() - s0)
            us = sorted(ts)[REPS // 2] * 1e6
            rows.append({"case": label, "grid": f"{H}x{W}", "n_modes": m, "engine": name,
                         "us_per_call": us, "status": "ok"})
            if us < best[1]:
                best = (name, us)
        except Exception as exc:
            rows.append({"case": label, "grid": f"{H}x{W}", "n_modes": m, "engine": name,
                         "us_per_call": float("nan"), "status": type(exc).__name__})
    return best[0], best[1], time.perf_counter() - t0


def build(C, H, W, m, B, dt):
    """Weights are an ARGUMENT, not a captured constant: closing over them bakes every
    parameter into the program as a literal and XLA then re-processes megabytes of
    constants for each shape. They are random, since only speed is measured."""
    P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, channels=C, n_modes=m, n_layers=4)
    x = jnp.asarray(np.random.default_rng(0).standard_normal((B, 1, H, W)).astype(np.float32))
    return jax.jit(lambda p, z: model.forward(p, z, m, dt)), (P, x)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"mode steps +-{MODE_STEPS}, batch steps +-{BATCH_STEPS}, "
          f"{'serial' if SERIAL else f'{WORKERS} compile threads'}\n", flush=True)
    rows, t_all = [], time.perf_counter()
    engine_rows = []
    for label, C, (H, W), m_pred in CASES:
        eng, eng_us, eng_t = pick_col_engine(label, C, H, W, m_pred, engine_rows)
        with open(ENGINE_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=ENGINE_FIELDS)
            w.writeheader(); w.writerows(engine_rows)
        print(f"### {label}  column DFT: {eng} at {eng_us:.2f} us/call "
              f"({eng_t:.1f}s to decide)", flush=True)
        b_pred = predict_batch(C, H, W)
        ms = [m for m in neighbours_add(m_pred, 8, MODE_STEPS) if m <= min(H, W)]
        bs = neighbours_mul(b_pred, BATCH_STEPS, lo=1)
        combos = [(m, B, dt, dn) for dt, dn in ((F, "float32"), (BF, "bfloat16"))
                  for m in ms for B in bs]
        print(f"### {label}  predicted m={m_pred} B={b_pred}   "
              f"m {ms} x B {bs} = {len(combos)} shapes", flush=True)

        t0 = time.perf_counter()
        built = [build(C, H, W, m, B, dt) for m, B, dt, _ in combos]

        def compile_one(t):
            # A shape too large to compile must not take the rest of the sweep with it.
            try:
                return t[0].lower(*t[1]).compile()
            except Exception as exc:
                return exc

        if SERIAL:
            compiled = [compile_one(t) for t in built]
        else:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                compiled = list(ex.map(compile_one, built))
        t_compile = time.perf_counter() - t0

        t0 = time.perf_counter()
        best = (None, None, None, 1e18)
        for (m, B, dt, dn), exe, (_, args) in zip(combos, compiled, built):
            if isinstance(exe, Exception):
                rows.append({"case": label, "dtype": dn, "n_modes": m, "batch": B,
                             "us_per_sample": float("nan"), "status": type(exe).__name__})
                continue
            try:
                jax.block_until_ready(exe(*args))
                ts = []
                for _ in range(REPS):
                    s = time.perf_counter(); jax.block_until_ready(exe(*args))
                    ts.append(time.perf_counter() - s)
                us = sorted(ts)[REPS // 2] * 1e6 / B
                if us < best[3]: best = (m, B, dn, us)
                rows.append({"case": label, "dtype": dn, "n_modes": m, "batch": B,
                             "us_per_sample": us, "status": "ok"})
            except Exception as exc:
                rows.append({"case": label, "dtype": dn, "n_modes": m, "batch": B,
                             "us_per_sample": float("nan"), "status": type(exc).__name__})
        t_run = time.perf_counter() - t0
        save(rows)
        print(f"    best m={best[0]} B={best[1]} {best[2]} {best[3]:.2f} us/sample", flush=True)
        print(f"    compile {t_compile:.1f}s   run {t_run:.1f}s   total {t_compile+t_run:.1f}s\n",
              flush=True)
    save(rows)
    print(f"TUNER2_DONE total {time.perf_counter()-t_all:.1f} s", flush=True)


if __name__ == "__main__":
    main()
