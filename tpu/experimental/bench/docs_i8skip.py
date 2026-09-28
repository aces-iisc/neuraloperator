"""int8 transforms at every grid, and three ways of paying for the skip path."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F, I8
from ...fno import layers
from ...fno.spectral import _E, _dft_rows, spectral_conv

CSV = os.environ.get("DOCS_I8_CSV", "/tmp/docs_i8skip.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_i8skip")
REPS = 20
FIELDS = ["case", "variant", "batch", "us_per_sample", "rel_err", "status", "trace"]

CASES = [("Burgers", 24, 16, (17, 16), 256), ("Darcy-32", 24, 16, (32, 32), 256),
         ("Darcy-64", 24, 16, (64, 64), 128), ("NS-128", 64, 64, (128, 128), 8)]


def band_of(n_modes, n_rows):
    half = n_modes // 2
    return np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)]), half


def _q8(a):
    s = jnp.max(jnp.abs(a)) / 127.0
    return jnp.round(a / s).astype(I8), s


def _mm8(spec, mat_q, z_q, scale):
    """int8 x int8 matmul with int32 accumulation, returned in f32 at the joint scale.

    The result is cast out of int32 immediately: leaving it live keeps a 4-byte buffer
    alive across the fusion, which is what exhausts the scoped VMEM budget at 128^2."""
    return jnp.einsum(spec, mat_q, z_q, preferred_element_type=jnp.int32).astype(F) * scale


def conv_int8(weight, x, n_modes):
    """Both transforms in int8; the learned channel mix stays f32."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    n_band, n_ch = band.shape[0], weight["w_re"].shape[1]

    row_cos, row_sin = _dft_rows(band, n_rows, -1)
    rq, rs = _q8(jnp.concatenate([row_cos, row_sin], axis=0))
    xq, xs = _q8(x)
    rows = _mm8("ah,nchw->ncaw", rq, xq, rs * xs)
    rows_re, rows_im = rows[:, :, :n_band], rows[:, :, n_band:]

    col_cos, col_sin = _dft_rows(np.arange(half), n_cols, -1)
    cq, cs = _q8(jnp.concatenate([col_cos, col_sin], axis=0))

    def over(z):
        zq, zs = _q8(z)
        return _mm8("dw,ncaw->ncad", cq, zq, cs * zs)

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
    mq, ms = _q8(jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cols_back = _mm8("wd,noad->noaw", iq, mq, isc * ms)
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]

    irc, irs = _dft_rows(band, n_rows, +1)
    vq, vs = _q8(jnp.concatenate([irc, -irs], axis=0).T)
    sq, ss = _q8(jnp.concatenate([cb_re, cb_im], axis=2))
    field = _mm8("ha,noaw->nohw", vq, sq, vs * ss)
    return (field / (n_rows * n_cols)).astype(F)


def conv_int8_rows_only(weight, x, n_modes):
    """int8 on the row pass only; the column pass and the inverse stay bfloat16."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    n_band, n_ch = band.shape[0], weight["w_re"].shape[1]

    row_cos, row_sin = _dft_rows(band, n_rows, -1)
    rq, rs = _q8(jnp.concatenate([row_cos, row_sin], axis=0))
    xq, xs = _q8(x)
    rows = _mm8("ah,nchw->ncaw", rq, xq, rs * xs).astype(BF)
    rows_re, rows_im = rows[:, :, :n_band], rows[:, :, n_band:]

    col_cos, col_sin = (c.astype(BF) for c in _dft_rows(np.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col_both, z)
    from_re, from_im = over(rows_re), over(rows_im)
    modes_re = from_re[..., :half] - from_im[..., half:]
    modes_im = from_re[..., half:] + from_im[..., :half]

    w_re, w_im = weight["w_re"].astype(BF), weight["w_im"].astype(BF)
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1)
    mixed = _E("ioad,niad->noad", w_block,
               jnp.concatenate([modes_re, modes_im], axis=1).astype(BF))
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]

    fold = jnp.where(jnp.arange(half) == 0, 1.0, 2.0).astype(F)
    icc, ics = _dft_rows(np.arange(half), n_cols, +1)
    icc, ics = (icc * fold[:, None]).T.astype(BF), (ics * fold[:, None]).T.astype(BF)
    inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                               jnp.concatenate([ics, icc], axis=1)], axis=0)
    cols_back = _E("wd,noad->noaw", inv_col,
                   jnp.concatenate([mixed_re, mixed_im], axis=-1).astype(BF))
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]
    irc, irs = (m.astype(BF) for m in _dft_rows(band, n_rows, +1))
    inv_row = jnp.concatenate([irc, -irs], axis=0)
    field = _E("ha,noaw->nohw", inv_row.T, jnp.concatenate([cb_re, cb_im], axis=2))
    return (field / (n_rows * n_cols)).astype(F)


# ---- the skip path -----------------------------------------------------------------

def layer_base(weight, skip, x, n_modes):
    return layers.gelu(spectral_conv(weight, x, n_modes) + layers.conv1x1(skip, x))


def layer_skip_bf16(weight, skip, x, n_modes):
    """Same layer, but the skip matmul carries bf16 operands and accumulates f32."""
    xb = x.astype(BF)
    s = jnp.einsum("oi,bihw->bohw", skip["W"].astype(BF), xb,
                   preferred_element_type=F) + skip["b"][None, :, None, None]
    return layers.gelu(spectral_conv(weight, x, n_modes) + s)


def layer_skip_in_matmul(weight, skip, x, n_modes):
    """Fold the add into the final inverse matmul: [inv_row | I] against [cols ; skip]."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    band, half = band_of(n_modes, n_rows)
    n_band, n_ch = band.shape[0], weight["w_re"].shape[1]
    dtype = F

    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(band, n_rows, -1))
    rows = _E("ah,nchw->ncaw", jnp.concatenate([row_cos, row_sin], axis=0), x)
    rows_re, rows_im = rows[:, :, :n_band], rows[:, :, n_band:]
    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    over = lambda z: _E("dw,ncaw->ncad", col_both, z)
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
    cols_back = _E("wd,noad->noaw", inv_col, jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cb_re, cb_im = cols_back[..., :n_cols], cols_back[..., n_cols:]

    irc, irs = _dft_rows(band, n_rows, +1)
    inv_row = (jnp.concatenate([irc, -irs], axis=0).T) / (n_rows * n_cols)
    combined = jnp.concatenate([inv_row, jnp.eye(n_rows, dtype=F)], axis=1)
    sk = jnp.einsum("oi,bihw->bohw", skip["W"], x, preferred_element_type=F)
    stacked = jnp.concatenate([cb_re, cb_im, sk], axis=2)
    field = _E("ha,noaw->nohw", combined, stacked)
    return layers.gelu(field + skip["b"][None, :, None, None])


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision = highest\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, C, m, (H, W), B in CASES:
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        sc = 1.0 / (C * C)
        weight = {"w_re": jax.random.normal(k1, (C, C, m, m // 2), F) * sc,
                  "w_im": jax.random.normal(k2, (C, C, m, m // 2), F) * sc}
        skip = {"W": jax.random.normal(k3, (C, C), F) * sc, "b": jnp.zeros((C,))}
        x = jax.random.normal(jax.random.PRNGKey(3), (B, C, H, W), F)

        conv_ref = np.asarray(spectral_conv(weight, x, m))
        layer_ref = np.asarray(layer_base(weight, skip, x, m))
        variants = {
            "conv f32":            (lambda z: spectral_conv(weight, z, m), conv_ref),
            "conv bf16":           (lambda z: spectral_conv(weight, z, m, BF), conv_ref),
            "conv int8":           (lambda z: conv_int8(weight, z, m), conv_ref),
            "conv int8 rows+bf16": (lambda z: conv_int8_rows_only(weight, z, m), conv_ref),
            "layer base":          (lambda z: layer_base(weight, skip, z, m), layer_ref),
            "layer skip bf16":     (lambda z: layer_skip_bf16(weight, skip, z, m), layer_ref),
            "layer skip in matmul":(lambda z: layer_skip_in_matmul(weight, skip, z, m), layer_ref),
        }
        print(f"--- {label}: C={C}, n_modes={m}, grid {H}x{W}, B={B}", flush=True)
        for name, (fn, ref) in variants.items():
            tag = f"{label}_{name.replace(' ','_').replace('+','')}"
            try:
                f = jax.jit(fn)
                got = np.asarray(jax.block_until_ready(f(x)))
                den = float(np.sqrt(np.sum(ref ** 2)))
                err = float(np.sqrt(np.sum((got - ref) ** 2))) / den
                ts = []
                with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        jax.block_until_ready(f(x))
                        ts.append(time.perf_counter() - t0)
                us = sorted(ts)[REPS // 2] * 1e6 / B
                print(f"    {name:<24}{us:>10.3f} us/sample   err {err:.2e}", flush=True)
                rows.append({"case": label, "variant": name, "batch": B, "us_per_sample": us,
                             "rel_err": err, "status": "ok", "trace": tag})
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"    {name:<24}{why:>10}", flush=True)
                print(f"        {str(exc)[:400]}", flush=True)
                rows.append({"case": label, "variant": name, "batch": B,
                             "us_per_sample": float("nan"), "rel_err": float("nan"),
                             "status": why, "trace": ""})
            save(rows)
        print(flush=True)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/i8skip.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("I8SKIP_DONE", flush=True)


if __name__ == "__main__":
    main()
