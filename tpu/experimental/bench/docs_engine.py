"""Which way of extracting the kept modes is fastest, swept over grid size and band width."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno.spectral import _E, _dft_rows

CSV = os.environ.get("DOCS_ENGINE_CSV", "/tmp/docs_engine.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_engine")
REPS = 20
CHANNELS = 24
BATCH = 128
FIELDS = ["n_grid", "n_modes", "variant", "m_rows", "k_rows", "us_per_sample",
          "status", "trace"]

# powers of two, sizes with a small odd factor, and two primes that cannot be staged at all
GRIDS = (16, 17, 24, 32, 48, 64, 96, 128)
MODES = (8, 16, 32)


def row_band_of(n_modes, n_rows):
    half = n_modes // 2
    return jnp.concatenate([jnp.arange(half), jnp.arange(n_rows - half, n_rows)]), half


def modes_partial(x, n_modes, dtype=F):
    """Only the kept band is ever computed: the shipping path."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    row_band, half = row_band_of(n_modes, n_rows)
    band = row_band.shape[0]
    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
    rows_both = _E("ah,nchw->ncaw", jnp.concatenate([row_cos, row_sin], axis=0), x.astype(dtype))
    rows_re, rows_im = rows_both[:, :, :band], rows_both[:, :, band:]
    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(jnp.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over_cols = lambda z: _E("dw,ncaw->ncad", col_both, z)
    from_re, from_im = over_cols(rows_re), over_cols(rows_im)
    return (from_re[..., :half] - from_im[..., half:],
            from_re[..., half:] + from_im[..., :half])


def modes_full(x, n_modes, dtype=F):
    """Every frequency is computed and the band sliced out afterwards: bigger tiles, more flops."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    row_band, half = row_band_of(n_modes, n_rows)
    all_rows, all_cols = jnp.arange(n_rows), jnp.arange(n_cols)
    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(all_rows, n_rows, -1))
    rows_both = _E("ah,nchw->ncaw", jnp.concatenate([row_cos, row_sin], axis=0), x.astype(dtype))
    rows_re = jnp.take(rows_both[:, :, :n_rows], row_band, axis=2)
    rows_im = jnp.take(rows_both[:, :, n_rows:], row_band, axis=2)
    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(all_cols, n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over_cols = lambda z: _E("dw,ncaw->ncad", col_both, z)
    from_re, from_im = over_cols(rows_re), over_cols(rows_im)
    re_cos, re_sin = from_re[..., :n_cols][..., :half], from_re[..., n_cols:][..., :half]
    im_cos, im_sin = from_im[..., :n_cols][..., :half], from_im[..., n_cols:][..., :half]
    return re_cos - im_sin, re_sin + im_cos


VARIANTS = {"partial": modes_partial, "full": modes_full}


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = highest, C={CHANNELS}, B={BATCH}\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for n_grid in GRIDS:
        for n_modes in MODES:
            if n_modes > n_grid:
                continue
            band, half = n_modes, n_modes // 2
            x = jax.random.normal(jax.random.PRNGKey(0),
                                  (BATCH, CHANNELS, n_grid, n_grid), F)
            ref = None
            print(f"--- grid {n_grid}x{n_grid}, n_modes {n_modes} "
                  f"(keeps {band}/{n_grid} rows)", flush=True)
            for name, fn in VARIANTS.items():
                tag = f"g{n_grid}_m{n_modes}_{name}"
                m_rows = 2 * band if name == "partial" else 2 * n_grid
                try:
                    f = jax.jit(lambda z, _f=fn, _m=n_modes: _f(z, _m))
                    got = jax.block_until_ready(f(x))
                    if ref is None:
                        ref = [np.asarray(g) for g in got]
                        err = 0.0
                    else:
                        den = sum(float(np.sqrt(np.sum(r ** 2))) for r in ref)
                        err = sum(float(np.sqrt(np.sum((np.asarray(g) - r) ** 2)))
                                  for g, r in zip(got, ref)) / den
                    ts = []
                    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                        for _ in range(REPS):
                            t0 = time.perf_counter()
                            jax.block_until_ready(f(x))
                            ts.append(time.perf_counter() - t0)
                    us = sorted(ts)[REPS // 2] * 1e6 / BATCH
                    print(f"    {name:<9} rows {m_rows:>4}x{n_grid:<4} "
                          f"{us:>9.3f} us/sample   err {err:.2e}", flush=True)
                    rows.append({"n_grid": n_grid, "n_modes": n_modes, "variant": name,
                                 "m_rows": m_rows, "k_rows": n_grid, "us_per_sample": us,
                                 "status": "ok", "trace": tag})
                except Exception as exc:
                    why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    print(f"    {name:<9} {why}", flush=True)
                    rows.append({"n_grid": n_grid, "n_modes": n_modes, "variant": name,
                                 "m_rows": m_rows, "k_rows": n_grid,
                                 "us_per_sample": float("nan"), "status": why, "trace": ""})
                save(rows)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/engine.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("\nENGINE_DONE", flush=True)


if __name__ == "__main__":
    main()
