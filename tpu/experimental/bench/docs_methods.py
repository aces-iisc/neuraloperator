"""Three transform methods behind the same optimized GEMM, swept over grid size and batch."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import ACC, BF, F
from ...fno import layers
from ...fno.spectral import _E, _dft_rows, spectral_conv

CSV = os.environ.get("DOCS_M_CSV", "/tmp/docs_methods.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_methods")
REPS = 20
FIELDS = ["case", "grid", "channels", "n_modes", "method", "batch",
          "us_per_sample", "rel_err", "status", "trace"]

# (label, grid, channels, n_modes)
CASES = [("N16", 16, 24, 16), ("N32", 32, 24, 16), ("N64", 64, 24, 16),
         ("N128", 128, 24, 16), ("NS128", 128, 64, 64)]
BATCHES = (8, 32, 128, 256)


def band_of(n_modes, n_rows):
    half = n_modes // 2
    return np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)]), half


# ---------------------------------------------------------------- row passes --

def rows_partial(x, n_modes, dtype):
    """Only the kept band, cosine and sine stacked into one matmul."""
    n_rows = x.shape[-2]
    band, _ = band_of(n_modes, n_rows)
    n_band = band.shape[0]
    cos, sin = (m.astype(dtype) for m in _dft_rows(band, n_rows, -1))
    both = _E("ah,nchw->ncaw", jnp.concatenate([cos, sin], axis=0), x.astype(dtype))
    return both[:, :, :n_band], both[:, :, n_band:]


def rows_coupled(x, n_modes, dtype):
    """Channel c pairs with c + C/2 into one complex signal, so half as many go through."""
    n_rows, n_ch = x.shape[-2], x.shape[1]
    band, half = band_of(n_modes, n_rows)
    n_band = band.shape[0]
    rows = np.concatenate([band, [half]])            # +1 partner the unpack needs
    cos, sin = (m.astype(dtype) for m in _dft_rows(rows, n_rows, -1))
    a = x[:, :n_ch // 2].astype(dtype)
    b = x[:, n_ch // 2:].astype(dtype)
    mix = lambda mat, z: _E("ah,nchw->ncaw", mat, z)
    p1, p2 = mix(cos, a), mix(sin, b)                 # Karatsuba: three matmuls, not four
    p3 = mix((cos + sin).astype(dtype), (a + b).astype(dtype))
    z_re, z_im = p1 - p2, p3 - p1 - p2
    where = {int(r): i for i, r in enumerate(rows)}
    mirror = jnp.asarray([where[int((n_rows - k) % n_rows)] for k in band])
    m_re = jnp.take(z_re, mirror, axis=2)
    m_im = jnp.take(z_im, mirror, axis=2)
    z_re, z_im = z_re[:, :, :n_band], z_im[:, :, :n_band]
    a_re, a_im = 0.5 * (z_re + m_re), 0.5 * (z_im - m_im)
    b_re, b_im = 0.5 * (z_im + m_im), -0.5 * (z_re - m_re)
    return (jnp.concatenate([a_re, b_re], axis=1),
            jnp.concatenate([a_im, b_im], axis=1))


def rows_reduced(x, n_modes, dtype):
    """Even and odd samples pack into one complex signal of half the length."""
    n_rows = x.shape[-2]
    band, _ = band_of(n_modes, n_rows)
    h = n_rows // 2
    x2 = x.astype(dtype).reshape(x.shape[0], x.shape[1], h, 2, x.shape[-1])
    even, odd = x2[:, :, :, 0, :], x2[:, :, :, 1, :]
    need = np.unique(np.concatenate([band % h, (-band) % h]))
    cos, sin = (m.astype(dtype) for m in _dft_rows(need, h, -1))
    mix = lambda mat, z: _E("ah,nchw->ncaw", mat, z)
    p1, p2 = mix(cos, even), mix(sin, odd)
    p3 = mix((cos + sin).astype(dtype), (even + odd).astype(dtype))
    z_re, z_im = p1 - p2, p3 - p1 - p2
    pos = {int(r): i for i, r in enumerate(need)}
    idx = jnp.asarray([pos[int(k % h)] for k in band])
    mir = jnp.asarray([pos[int((-k) % h)] for k in band])
    zr, zi = jnp.take(z_re, idx, axis=2), jnp.take(z_im, idx, axis=2)
    mr, mi = jnp.take(z_re, mir, axis=2), -jnp.take(z_im, mir, axis=2)
    e_re, e_im = 0.5 * (zr + mr), 0.5 * (zi + mi)
    o_re, o_im = 0.5 * (zi - mi), -0.5 * (zr - mr)
    ang = (-2.0 * jnp.pi / n_rows) * jnp.asarray(band).astype(F)
    tw_r, tw_i = jnp.cos(ang)[None, None, :, None], jnp.sin(ang)[None, None, :, None]
    return (e_re + (tw_r * o_re - tw_i * o_im),
            e_im + (tw_r * o_im + tw_i * o_re))


ROWS = {"partial": rows_partial, "coupled": rows_coupled, "reduced": rows_reduced}


def conv(rows_fn, w_block, x, n_modes, n_ch, dtype=F):
    """Rows by the chosen method; columns, the blocked ADIO GEMM and the inverse are shared."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    rows_re, rows_im = rows_fn(x, n_modes, dtype)

    cos, sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
    col = jnp.concatenate([cos, sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col, z)
    from_re, from_im = over(rows_re), over(rows_im)
    modes_re = from_re[..., :half] - from_im[..., half:]
    modes_im = from_re[..., half:] + from_im[..., :half]

    z_block = jnp.concatenate([modes_re, modes_im], axis=1)
    mixed = _E("adio,niad->noad", w_block.astype(dtype), z_block)
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    fold = jnp.where(jnp.arange(half) == 0, 1.0, 2.0).astype(F)
    icc, ics = _dft_rows(np.arange(half), n_cols, +1)
    icc, ics = (icc * fold[:, None]).T, (ics * fold[:, None]).T
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    cols_back = _E("wd,noad->noaw", inv_col, jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]
    irc, irs = _dft_rows(band, n_rows, +1)
    inv_row = jnp.concatenate([irc, -irs], axis=0)
    field = _E("ha,noaw->nohw", inv_row.T, jnp.concatenate([cb_re, cb_im], axis=2))
    return (field / (n_rows * n_cols)).astype(F)


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision = highest; GEMM is blocked + ADIO + prebuilt for every method\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, N, C, m in CASES:
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        sc = 1.0 / (C * C)
        shape = (m, m // 2, C, C)                       # ADIO layout
        w_re = jax.random.normal(k1, shape, F) * sc
        w_im = jax.random.normal(k2, shape, F) * sc
        w_block = layers.spectral_block(w_re, w_im, F)  # hoisted: built once, passed in
        ref = None
        print(f"--- {label}: grid {N}x{N}, C={C}, n_modes={m}", flush=True)
        for method, fn in ROWS.items():
            if method == "coupled" and C % 2:
                continue
            if method == "reduced" and N % 2:
                continue
            for B in BATCHES:
                tag = f"{label}_{method}_{B}"
                x = jax.random.normal(jax.random.PRNGKey(3), (B, C, N, N), F)
                try:
                    f = jax.jit(lambda z, _f=fn: conv(_f, w_block, z, m, C, F))
                    got = np.asarray(jax.block_until_ready(f(x)))
                    if method == "partial" and B == BATCHES[0]:
                        ref = got
                        err = 0.0
                    elif ref is not None and got.shape[0] >= ref.shape[0]:
                        n = ref.shape[0]
                        den = float(np.sqrt(np.sum(ref ** 2)))
                        err = float(np.sqrt(np.sum((got[:n] - ref) ** 2))) / den
                    else:
                        err = float("nan")
                    ts = []
                    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                        for _ in range(REPS):
                            t0 = time.perf_counter()
                            jax.block_until_ready(f(x))
                            ts.append(time.perf_counter() - t0)
                    us = sorted(ts)[REPS // 2] * 1e6 / B
                    print(f"    {method:<9}B={B:<5}{us:>9.3f} us/sample   err {err:.2e}",
                          flush=True)
                    rows.append({"case": label, "grid": N, "channels": C, "n_modes": m,
                                 "method": method, "batch": B, "us_per_sample": us,
                                 "rel_err": err, "status": "ok", "trace": tag})
                except Exception as exc:
                    why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    print(f"    {method:<9}B={B:<5}{why:>9}", flush=True)
                    rows.append({"case": label, "grid": N, "channels": C, "n_modes": m,
                                 "method": method, "batch": B, "us_per_sample": float("nan"),
                                 "rel_err": float("nan"), "status": why, "trace": ""})
                save(rows)
        print(flush=True)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/methods.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("METHODS_DONE", flush=True)


if __name__ == "__main__":
    main()
