"""Four traffic-side variants of the Fourier layer, against the shipping one."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F, I8
from ...fno import layers
from ...fno.spectral import _E, _dft_rows, spectral_conv

CSV = os.environ.get("DOCS_OPT_CSV", "/tmp/docs_opt.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_opt")
REPS = 20
FIELDS = ["case", "variant", "batch", "us_per_sample", "rel_err", "status", "trace"]

# (label, channels, n_modes, grid, batch)
CASES = [("Burgers", 24, 16, (17, 16), 256), ("Darcy-32", 24, 16, (32, 32), 256),
         ("Darcy-64", 24, 16, (64, 64), 128), ("NS-128", 64, 64, (128, 128), 8)]


def band_of(n_modes, n_rows):
    half = n_modes // 2
    return np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)]), half


def conv_layout(weight, x, n_modes, dtype=F):
    """Same maths, but every contraction is made to fall on the minor axis."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    n_band, n_ch = band.shape[0], weight["w_re"].shape[1]

    xt = jnp.transpose(x.astype(dtype), (0, 1, 3, 2))                  # (n,c,w,h), h minor
    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(band, n_rows, -1))
    rows = _E("ah,ncwh->ncwa", jnp.concatenate([row_cos, row_sin], axis=0), xt)
    rows_re = jnp.transpose(rows[..., :n_band], (0, 1, 3, 2))          # (n,c,a,w), w minor
    rows_im = jnp.transpose(rows[..., n_band:], (0, 1, 3, 2))

    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col_both, z)
    from_re, from_im = over(rows_re), over(rows_im)
    modes_re = from_re[..., :half] - from_im[..., half:]
    modes_im = from_re[..., half:] + from_im[..., :half]

    w_re, w_im = weight["w_re"].astype(dtype), weight["w_im"].astype(dtype)
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1)
    z_block = jnp.concatenate([modes_re, modes_im], axis=1)
    mixed = _E("ioad,niad->noad", w_block, z_block)
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    col_modes = np.arange(half)
    fold = jnp.where(jnp.asarray(col_modes) == 0, 1.0, 2.0).astype(F)
    icc, ics = _dft_rows(col_modes, n_cols, +1)
    icc, ics = (icc * fold[:, None]).T, (ics * fold[:, None]).T
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    cols_back = _E("wd,noad->noaw", inv_col, jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]

    irc, irs = _dft_rows(band, n_rows, +1)
    inv_row = jnp.concatenate([irc, -irs], axis=0)
    stacked = jnp.concatenate([cb_re, cb_im], axis=2)                  # (n,o,2a,w)
    st = jnp.transpose(stacked, (0, 1, 3, 2))                          # (n,o,w,2a), a minor
    field = _E("ha,nowa->nowh", inv_row.T, st)
    field = jnp.transpose(field, (0, 1, 3, 2))
    return (field / (n_rows * n_cols)).astype(F)


def conv_hoist(weight, x, n_modes, dtype=F):
    """The Hermitian fold moved off the inverse matrix and onto the learned weight."""
    half = n_modes // 2
    fold = jnp.where(jnp.arange(half) == 0, 1.0, 2.0).astype(F)
    folded = {"w_re": weight["w_re"] * fold, "w_im": weight["w_im"] * fold}
    return _conv_nofold(folded, x, n_modes, dtype)


def _conv_nofold(weight, x, n_modes, dtype=F):
    """spectral_conv with the fold removed from the inverse column matrix."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    n_band, n_ch = band.shape[0], weight["w_re"].shape[1]
    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(band, n_rows, -1))
    rows = _E("ah,nchw->ncaw", jnp.concatenate([row_cos, row_sin], axis=0), x.astype(dtype))
    rows_re, rows_im = rows[:, :, :n_band], rows[:, :, n_band:]
    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col_both, z)
    from_re, from_im = over(rows_re), over(rows_im)
    modes_re = from_re[..., :half] - from_im[..., half:]
    modes_im = from_re[..., half:] + from_im[..., :half]
    w_re, w_im = weight["w_re"].astype(dtype), weight["w_im"].astype(dtype)
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1)
    mixed = _E("ioad,niad->noad", w_block,
               jnp.concatenate([modes_re, modes_im], axis=1))
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]
    icc, ics = _dft_rows(np.arange(half), n_cols, +1)          # no fold here
    icc, ics = icc.T, ics.T
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    cols_back = _E("wd,noad->noaw", inv_col, jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]
    irc, irs = _dft_rows(band, n_rows, +1)
    inv_row = jnp.concatenate([irc, -irs], axis=0)
    field = _E("ha,noaw->nohw", inv_row.T, jnp.concatenate([cb_re, cb_im], axis=2))
    return (field / (n_rows * n_cols)).astype(F)


def _q8(a):
    """Symmetric per-tensor int8 quantisation, returning the codes and their scale."""
    s = jnp.max(jnp.abs(a)) / 127.0
    return jnp.round(a / s).astype(I8), s


def conv_int8(weight, x, n_modes, dtype=I8):
    """Transforms carried in int8 with int32 accumulation; the channel GEMM stays f32."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    n_band, n_ch = band.shape[0], weight["w_re"].shape[1]
    acc = lambda spec, m, z: jnp.einsum(spec, m, z, preferred_element_type=jnp.int32)

    row_cos, row_sin = _dft_rows(band, n_rows, -1)
    rq, rs = _q8(jnp.concatenate([row_cos, row_sin], axis=0))
    xq, xs = _q8(x)
    rows = acc("ah,nchw->ncaw", rq, xq).astype(F) * (rs * xs)
    rows_re, rows_im = rows[:, :, :n_band], rows[:, :, n_band:]

    col_cos, col_sin = _dft_rows(np.arange(half), n_cols, -1)
    cq, cs = _q8(jnp.concatenate([col_cos, col_sin], axis=0))
    def over(z):
        zq, zs = _q8(z)
        return acc("dw,ncaw->ncad", cq, zq).astype(F) * (cs * zs)
    from_re, from_im = over(rows_re), over(rows_im)
    modes_re = from_re[..., :half] - from_im[..., half:]
    modes_im = from_re[..., half:] + from_im[..., :half]

    w_re, w_im = weight["w_re"], weight["w_im"]
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1)
    mixed = _E("ioad,niad->noad", w_block, jnp.concatenate([modes_re, modes_im], axis=1))
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    fold = jnp.where(jnp.arange(half) == 0, 1.0, 2.0).astype(F)
    icc, ics = _dft_rows(np.arange(half), n_cols, +1)
    icc, ics = (icc * fold[:, None]).T, (ics * fold[:, None]).T
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    iq, isc = _q8(inv_col)
    mb = jnp.concatenate([mixed_re, mixed_im], axis=-1)
    mq, ms = _q8(mb)
    cols_back = acc("wd,noad->noaw", iq, mq).astype(F) * (isc * ms)
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]

    irc, irs = _dft_rows(band, n_rows, +1)
    inv_row = jnp.concatenate([irc, -irs], axis=0)
    vq, vs = _q8(inv_row.T)
    sb = jnp.concatenate([cb_re, cb_im], axis=2)
    sq, ss = _q8(sb)
    field = acc("ha,noaw->nohw", vq, sq).astype(F) * (vs * ss)
    return (field / (n_rows * n_cols)).astype(F)


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision = highest\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, C, m, (H, W), B in CASES:
        key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        scale = 1.0 / (C * C)
        weight = {"w_re": jax.random.normal(k1, (C, C, m, m // 2), F) * scale,
                  "w_im": jax.random.normal(k2, (C, C, m, m // 2), F) * scale}
        skip = {"W": jax.random.normal(k3, (C, C), F) * scale, "b": jnp.zeros((C,))}
        x = jax.random.normal(jax.random.PRNGKey(3), (B, C, H, W), F)

        # the full layer, and the pieces of it, to see whether the skip path is worth fusing
        full_layer = lambda p, z: layers.gelu(spectral_conv(p[0], z, m) + layers.conv1x1(p[1], z))
        variants = {
            "layer (spectral+skip+gelu)": lambda z: full_layer((weight, skip), z),
            "spectral only":              lambda z: spectral_conv(weight, z, m),
            "skip only":                  lambda z: layers.conv1x1(skip, z),
            "layer, fused add+gelu":      lambda z: layers.gelu(
                                              spectral_conv(weight, z, m) + layers.conv1x1(skip, z)),
            "conv base":                  lambda z: spectral_conv(weight, z, m),
            "conv layout":                lambda z: conv_layout(weight, z, m),
            "conv hoist-fold":            lambda z: conv_hoist(weight, z, m),
            "conv bf16":                  lambda z: spectral_conv(weight, z, m, BF),
            "conv int8":                  lambda z: conv_int8(weight, z, m),
        }
        ref = np.asarray(spectral_conv(weight, x, m))
        den = float(np.sqrt(np.sum(ref ** 2)))
        print(f"--- {label}: C={C}, n_modes={m}, grid {H}x{W}, B={B}", flush=True)
        for name, fn in variants.items():
            tag = f"{label}_{name.replace(' ', '_').replace('+','')}"
            try:
                f = jax.jit(fn)
                got = np.asarray(jax.block_until_ready(f(x)))
                err = (float(np.sqrt(np.sum((got - ref) ** 2))) / den
                       if got.shape == ref.shape and name.startswith("conv") else float("nan"))
                ts = []
                with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        jax.block_until_ready(f(x))
                        ts.append(time.perf_counter() - t0)
                us = sorted(ts)[REPS // 2] * 1e6 / B
                print(f"    {name:<28}{us:>10.3f} us/sample   err {err:.2e}", flush=True)
                rows.append({"case": label, "variant": name, "batch": B, "us_per_sample": us,
                             "rel_err": err, "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"    {name:<28}{why:>10}  {str(exc)[:80]}", flush=True)
                rows.append({"case": label, "variant": name, "batch": B,
                             "us_per_sample": float("nan"), "rel_err": float("nan"),
                             "status": why, "trace": ""})
            save(rows)
        print(flush=True)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/opt.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("OPT_DONE", flush=True)


if __name__ == "__main__":
    main()
