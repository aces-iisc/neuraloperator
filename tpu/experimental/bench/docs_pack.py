"""Coupled and reduced packings of the row transform, against the plain partial one."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno.spectral import _E, _dft_rows

CSV = os.environ.get("DOCS_PACK_CSV", "/tmp/docs_pack.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_pack")
REPS = 20
CHANNELS = 24
BATCH = 128
GRIDS = (16, 32, 64, 128)
N_MODES = 16
FIELDS = ["n_grid", "n_modes", "variant", "us_per_sample", "rel_err", "status", "trace"]


def row_band_of(n_modes, n_rows):
    """Static numpy indices: n_modes and n_rows are Python ints, so this must not be traced."""
    half = n_modes // 2
    return np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)]), half


def rows_partial(x, n_modes, dtype=F):
    """The shipping row transform: cos and sin stacked, real input, two matmuls in one."""
    n_rows = x.shape[-2]
    row_band, _ = row_band_of(n_modes, n_rows)
    band = row_band.shape[0]
    cos, sin = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
    both = _E("ah,nchw->ncaw", jnp.concatenate([cos, sin], axis=0), x.astype(dtype))
    return both[:, :, :band], both[:, :, band:]


def rows_coupled(x, n_modes, dtype=F):
    """Pair the channels into complex signals, transform once, then separate them again.

    Channel c and channel c + C/2 become z = a + i b, so half as many signals go through the
    matmul.  Recovering a and b needs Z at both +k and -k; the kept band already holds those
    pairs, except that the partner of row N - half is +half, which has to be added."""
    n_rows, n_ch = x.shape[-2], x.shape[1]
    row_band, half = row_band_of(n_modes, n_rows)
    band = row_band.shape[0]
    rows = np.concatenate([row_band, [half]])         # band + 1 rows of the complex DFT
    cos, sin = (m.astype(dtype) for m in _dft_rows(rows, n_rows, -1))

    a = x[:, :n_ch // 2].astype(dtype)
    b = x[:, n_ch // 2:].astype(dtype)
    mix = lambda mat, z: _E("ah,nchw->ncaw", mat, z)
    # Z = (cos + i sin)(a + i b) by Karatsuba: three matmuls instead of four
    p1, p2 = mix(cos, a), mix(sin, b)
    p3 = mix((cos + sin).astype(dtype), (a + b).astype(dtype))
    z_re, z_im = p1 - p2, p3 - p1 - p2

    # index of -k inside `rows`, for every k in the kept band
    where = {int(r): i for i, r in enumerate(rows)}
    mirror_idx = np.array([where[int((n_rows - k) % n_rows)] for k in row_band])
    m_re = jnp.take(z_re, jnp.asarray(mirror_idx), axis=2)
    m_im = jnp.take(z_im, jnp.asarray(mirror_idx), axis=2)
    z_re, z_im = z_re[:, :, :band], z_im[:, :, :band]

    a_re, a_im = 0.5 * (z_re + m_re), 0.5 * (z_im - m_im)      # A = (Z + conj(Z_-k)) / 2
    b_re, b_im = 0.5 * (z_im + m_im), -0.5 * (z_re - m_re)     # B = (Z - conj(Z_-k)) / 2i
    return (jnp.concatenate([a_re, b_re], axis=1),
            jnp.concatenate([a_im, b_im], axis=1))


def rows_reduced(x, n_modes, dtype=F):
    """Split each column into even and odd samples, pack them as one complex signal of half the
    length, transform that, then twist the two sub-transforms back together."""
    n_rows = x.shape[-2]
    row_band, _ = row_band_of(n_modes, n_rows)
    band, h = row_band.shape[0], n_rows // 2
    x2 = x.astype(dtype).reshape(x.shape[0], x.shape[1], h, 2, x.shape[-1])
    even, odd = x2[:, :, :, 0, :], x2[:, :, :, 1, :]

    need = np.unique(np.concatenate([row_band % h, (-row_band) % h]))
    sub = jnp.asarray(need)
    cos, sin = (m.astype(dtype) for m in _dft_rows(sub, h, -1))
    mix = lambda mat, z: _E("ah,nchw->ncaw", mat, z)
    p1, p2 = mix(cos, even), mix(sin, odd)
    p3 = mix((cos + sin).astype(dtype), (even + odd).astype(dtype))
    z_re, z_im = p1 - p2, p3 - p1 - p2

    pos = {int(r): i for i, r in enumerate(need)}
    idx = jnp.asarray([pos[int(k % h)] for k in row_band])
    mir = jnp.asarray([pos[int((-k) % h)] for k in row_band])
    zr, zi = jnp.take(z_re, idx, axis=2), jnp.take(z_im, idx, axis=2)
    mr, mi = jnp.take(z_re, mir, axis=2), -jnp.take(z_im, mir, axis=2)

    e_re, e_im = 0.5 * (zr + mr), 0.5 * (zi + mi)              # even sub-DFT
    o_re, o_im = 0.5 * (zi - mi), -0.5 * (zr - mr)             # odd sub-DFT
    ang = (-2.0 * jnp.pi / n_rows) * jnp.asarray(row_band).astype(F)
    tw_r, tw_i = jnp.cos(ang)[None, None, :, None], jnp.sin(ang)[None, None, :, None]
    return (e_re + (tw_r * o_re - tw_i * o_im),
            e_im + (tw_r * o_im + tw_i * o_re))


VARIANTS = {"partial": rows_partial, "coupled": rows_coupled, "reduced": rows_reduced}


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = highest, C={CHANNELS}, B={BATCH}, n_modes={N_MODES}\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for n_grid in GRIDS:
        x = jax.random.normal(jax.random.PRNGKey(0), (BATCH, CHANNELS, n_grid, n_grid), F)
        print(f"--- grid {n_grid}x{n_grid}", flush=True)
        ref = None
        for name, fn in VARIANTS.items():
            tag = f"g{n_grid}_{name}"
            try:
                f = jax.jit(lambda z, _f=fn: _f(z, N_MODES))
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
                print(f"    {name:<9}{us:>9.3f} us/sample   err {err:.2e}", flush=True)
                rows.append({"n_grid": n_grid, "n_modes": N_MODES, "variant": name,
                             "us_per_sample": us, "rel_err": err, "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"    {name:<9}{why:>9}  {str(exc)[:90]}", flush=True)
                rows.append({"n_grid": n_grid, "n_modes": N_MODES, "variant": name,
                             "us_per_sample": float("nan"), "rel_err": float("nan"),
                             "status": why, "trace": ""})
            save(rows)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/pack.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("\nPACK_DONE", flush=True)


if __name__ == "__main__":
    main()
