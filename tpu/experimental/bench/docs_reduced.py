"""Folding the even/odd selection into the DFT matrix, against the split-and-slice version."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import layers
from ...fno.spectral import _E, _dft_rows

CSV = os.environ.get("DOCS_R_CSV", "/tmp/docs_reduced.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_reduced")
REPS = 20
FIELDS = ["case", "grid", "channels", "n_modes", "method", "dtype", "batch",
          "us_per_sample", "rel_err", "status", "trace"]
CASES = [("Darcy-32", 32, 24, 16, (32, 128, 256)),
         ("Darcy-64", 64, 24, 16, (32, 128, 256)),
         ("NS-128", 128, 64, 64, (4, 8, 16))]


def band_of(n_modes, n_rows):
    half = n_modes // 2
    return np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)]), half


def _recombine(z_re, z_im, band, need, h, n_rows):
    """Shared tail: pull E and O out of the half-length spectrum and twist them back."""
    pos = {int(r): i for i, r in enumerate(need)}
    idx = jnp.asarray([pos[int(k % h)] for k in band])
    mir = jnp.asarray([pos[int((-k) % h)] for k in band])
    zr, zi = jnp.take(z_re, idx, axis=2), jnp.take(z_im, idx, axis=2)
    mr, mi = jnp.take(z_re, mir, axis=2), -jnp.take(z_im, mir, axis=2)
    e_re, e_im = 0.5 * (zr + mr), 0.5 * (zi + mi)
    o_re, o_im = 0.5 * (zi - mi), -0.5 * (zr - mr)
    ang = (-2.0 * jnp.pi / n_rows) * jnp.asarray(band).astype(F)
    tr, ti = jnp.cos(ang)[None, None, :, None], jnp.sin(ang)[None, None, :, None]
    return e_re + (tr * o_re - ti * o_im), e_im + (tr * o_im + ti * o_re)


def rows_reduced_split(x, n_modes, dtype):
    """As measured before: reshape to (h, 2, W), slice out even and odd, Karatsuba."""
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
    return _recombine(p1 - p2, p3 - p1 - p2, band, need, h, n_rows)


def rows_reduced_folded(x, n_modes, dtype):
    """The even/odd selection folded into the matrix, so it is the matmul's read pattern.

    even[p] = x[2p] and odd[p] = x[2p+1], so a matrix indexed by the FULL row n with
    p = n//2 reaches both streams without materialising either.  Real and imaginary
    parts stack into one matmul, as in the partial transform."""
    n_rows = x.shape[-2]
    band, _ = band_of(n_modes, n_rows)
    h = n_rows // 2
    need = np.unique(np.concatenate([band % h, (-band) % h]))
    n = np.arange(n_rows)
    even_col = (n % 2 == 0)
    ang = (-2.0 * np.pi / h) * np.outer(need, n // 2)
    c, s = np.cos(ang), np.sin(ang)
    m_re = np.where(even_col[None, :], c, -s)
    m_im = np.where(even_col[None, :], s, c)
    stacked = jnp.asarray(np.concatenate([m_re, m_im], axis=0)).astype(dtype)
    both = _E("ah,nchw->ncaw", stacked, x.astype(dtype))
    nn = need.shape[0]
    return _recombine(both[:, :, :nn], both[:, :, nn:], band, need, h, n_rows)


def rows_partial(x, n_modes, dtype):
    n_rows = x.shape[-2]
    band, _ = band_of(n_modes, n_rows)
    nb = band.shape[0]
    cos, sin = (m.astype(dtype) for m in _dft_rows(band, n_rows, -1))
    both = _E("ah,nchw->ncaw", jnp.concatenate([cos, sin], axis=0), x.astype(dtype))
    return both[:, :, :nb], both[:, :, nb:]


METHODS = {"partial": rows_partial, "reduced-split": rows_reduced_split,
           "reduced-folded": rows_reduced_folded}


def conv(rows_fn, w_block, x, n_modes, n_ch, dtype):
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    rows_re, rows_im = rows_fn(x, n_modes, dtype)
    cos, sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
    col = jnp.concatenate([cos, sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col, z)
    f_re, f_im = over(rows_re), over(rows_im)
    modes_re = f_re[..., :half] - f_im[..., half:]
    modes_im = f_re[..., half:] + f_im[..., :half]
    mixed = _E("adio,niad->noad", w_block.astype(dtype),
               jnp.concatenate([modes_re, modes_im], axis=1))
    m_re, m_im = mixed[:, :n_ch], mixed[:, n_ch:]
    fold = jnp.where(jnp.arange(half) == 0, 1.0, 2.0).astype(F)
    icc, ics = _dft_rows(np.arange(half), n_cols, +1)
    icc, ics = (icc * fold[:, None]).T, (ics * fold[:, None]).T
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    cb = _E("wd,noad->noaw", inv_col, jnp.concatenate([m_re, m_im], axis=-1))
    cbr, cbi = cb[..., :n_cols], cb[..., n_cols:]
    irc, irs = _dft_rows(band, n_rows, +1)
    field = _E("ha,noaw->nohw", jnp.concatenate([irc, -irs], axis=0).T,
               jnp.concatenate([cbr, cbi], axis=2))
    return (field / (n_rows * n_cols)).astype(F)


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, N, C, m, batches in CASES:
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        sc = 1.0 / (C * C)
        wr = jax.random.normal(k1, (m, m // 2, C, C), F) * sc
        wi = jax.random.normal(k2, (m, m // 2, C, C), F) * sc
        print(f"--- {label}: grid {N}x{N}, C={C}, n_modes={m}", flush=True)
        for dname, dt in (("f32", F), ("bf16", BF)):
            wb = layers.spectral_block(wr, wi, dt)
            ref = None
            for method, fn in METHODS.items():
                for B in batches:
                    tag = f"{label}_{method}_{dname}_{B}"
                    x = jax.random.normal(jax.random.PRNGKey(3), (B, C, N, N), F)
                    try:
                        f = jax.jit(lambda z, _f=fn: conv(_f, wb, z, m, C, dt))
                        got = np.asarray(jax.block_until_ready(f(x)))
                        if method == "partial" and B == batches[0]:
                            ref, err = got, 0.0
                        elif ref is not None and got.shape[0] >= ref.shape[0]:
                            k = ref.shape[0]
                            err = float(np.sqrt(np.sum((got[:k] - ref) ** 2))) / float(
                                np.sqrt(np.sum(ref ** 2)))
                        else:
                            err = float("nan")
                        ts = []
                        with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                            for _ in range(REPS):
                                t0 = time.perf_counter()
                                jax.block_until_ready(f(x))
                                ts.append(time.perf_counter() - t0)
                        us = sorted(ts)[REPS // 2] * 1e6 / B
                        print(f"    {method:<16}{dname:<6}B={B:<5}{us:>9.3f} us  "
                              f"err {err:.2e}", flush=True)
                        rows.append({"case": label, "grid": N, "channels": C, "n_modes": m,
                                     "method": method, "dtype": dname, "batch": B,
                                     "us_per_sample": us, "rel_err": err,
                                     "status": "ok", "trace": tag})
                    except Exception as exc:
                        why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                        print(f"    {method:<16}{dname:<6}B={B:<5}{why:>9}", flush=True)
                        rows.append({"case": label, "grid": N, "channels": C, "n_modes": m,
                                     "method": method, "dtype": dname, "batch": B,
                                     "us_per_sample": float("nan"), "rel_err": float("nan"),
                                     "status": why, "trace": ""})
                    save(rows)
        print(flush=True)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/reduced.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("REDUCED_DONE", flush=True)


if __name__ == "__main__":
    main()
