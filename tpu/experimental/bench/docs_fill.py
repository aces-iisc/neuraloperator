"""Splits FFT from GEMM inside the spectral conv, and times two ways of filling the MXU tile."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno.spectral import _E, _dft_rows, spectral_conv

CSV = os.environ.get("DOCS_FILL_CSV", "/tmp/docs_fill.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_fill")
REPS = 20
FIELDS = ["case", "variant", "batch", "us_per_sample", "status", "trace"]

# (label, channels, n_modes, grid)
CASES = [("Darcy-32", 24, 16, 32),
         ("Darcy-64", 24, 16, 64),
         ("NS-128", 64, 64, 128)]
BATCHES = (8, 16, 32, 64, 128)
FUSED_MAX_ENTRIES = 8 << 20      # the dense 2-D matrix is band*half*rows*cols; cap it


def row_band_of(n_modes, n_rows):
    half = n_modes // 2
    return jnp.concatenate([jnp.arange(half), jnp.arange(n_rows - half, n_rows)]), half


def forward_modes(x, n_modes, dtype=F):
    """Steps 1-2, separable: transform rows, then columns."""
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


def forward_modes_fused(x, n_modes, dtype=F):
    """Steps 1-2 as ONE dense 2-D DFT matmul: more flops, but every MXU lane is live."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    row_band, half = row_band_of(n_modes, n_rows)
    band = row_band.shape[0]
    rows = jnp.arange(n_rows)
    cols = jnp.arange(n_cols)
    # theta = -2*pi*(k*h/n_rows + d*w/n_cols), kept on integers until the last divide
    phase = (row_band[:, None, None, None] * rows[None, None, :, None] * n_cols
             + jnp.arange(half)[None, :, None, None] * cols[None, None, None, :] * n_rows)
    angle = (-2.0 * jnp.pi / (n_rows * n_cols)) * phase.astype(F)
    n_mode = band * half
    mat = jnp.concatenate([jnp.cos(angle).reshape(n_mode, n_rows * n_cols),
                           jnp.sin(angle).reshape(n_mode, n_rows * n_cols)], axis=0).astype(dtype)
    flat = x.astype(dtype).reshape(x.shape[0], x.shape[1], n_rows * n_cols)
    out = _E("mk,nck->ncm", mat, flat)
    shape = (out.shape[0], out.shape[1], band, half)
    return out[..., :n_mode].reshape(shape), out[..., n_mode:].reshape(shape)


def forward_modes_stackcols(x, n_modes, dtype=F):
    """Steps 1-2, but the column transform hits re and im in ONE call: same flops, one matmul."""
    n_rows, n_cols = x.shape[-2], x.shape[-1]
    row_band, half = row_band_of(n_modes, n_rows)
    band = row_band.shape[0]
    n_ch = x.shape[1]
    row_cos, row_sin = (m.astype(dtype) for m in _dft_rows(row_band, n_rows, -1))
    rows_both = _E("ah,nchw->ncaw", jnp.concatenate([row_cos, row_sin], axis=0), x.astype(dtype))
    rows_re, rows_im = rows_both[:, :, :band], rows_both[:, :, band:]
    col_cos, col_sin = (c.astype(dtype) for c in _dft_rows(jnp.arange(half), n_cols, -1))
    col_both = jnp.concatenate([col_cos, col_sin], axis=0)
    stacked = jnp.concatenate([rows_re, rows_im], axis=1)          # share the matrix, widen N
    out = _E("dw,ncaw->ncad", col_both, stacked)
    from_re, from_im = out[:, :n_ch], out[:, n_ch:]
    return (from_re[..., :half] - from_im[..., half:],
            from_re[..., half:] + from_im[..., :half])


def gemm_blocked(w_re, w_im, modes_re, modes_im, dtype=F):
    """Step 3 as it ships: one real block matmul per mode."""
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1).astype(dtype)
    z = jnp.concatenate([modes_re, modes_im], axis=1).astype(dtype)
    return _E("ioad,niad->noad", w_block, z)


def gemm_grouped(w_re, w_im, modes_re, modes_im, group, dtype=F):
    """Step 3 with `group` modes sharing one block-diagonal matmul, so the tile is fuller."""
    w_block = jnp.concatenate([jnp.concatenate([w_re, -w_im], axis=0),
                               jnp.concatenate([w_im, w_re], axis=0)], axis=1).astype(dtype)
    z = jnp.concatenate([modes_re, modes_im], axis=1).astype(dtype)
    n, two_c, band, half = z.shape
    n_mode = band * half
    n_blk = n_mode // group
    flat_w = w_block.reshape(two_c, two_c, n_mode)
    stripes = []
    for slot in range(group):
        live = flat_w[:, :, slot::group]
        stripes.append(jnp.concatenate(
            [live if other == slot else jnp.zeros_like(live) for other in range(group)], axis=1))
    block_diag = jnp.concatenate(stripes, axis=0)            # (two_c*g, two_c*g, n_blk)
    zg = (z.reshape(n, two_c, n_blk, group).transpose(0, 3, 1, 2)
           .reshape(n, two_c * group, n_blk))
    og = _E("iob,nib->nob", block_diag, zg)
    return (og.reshape(n, group, two_c, n_blk).transpose(0, 2, 3, 1)
              .reshape(n, two_c, band, half))


def inverse_field(mixed, n_modes, n_rows, n_cols, n_ch):
    """Steps 4a-4b: the kept modes back to a real field."""
    half = n_modes // 2
    mixed_re, mixed_im = mixed[:, :n_ch], mixed[:, n_ch:]
    col_modes = jnp.arange(half)
    fold = jnp.where(col_modes == 0, 1.0, 2.0).astype(F)
    inv_col_cos, inv_col_sin = _dft_rows(col_modes, n_cols, +1)
    inv_col_cos = (inv_col_cos * fold[:, None]).T
    inv_col_sin = (inv_col_sin * fold[:, None]).T
    inv_col_block = jnp.concatenate(
        [jnp.concatenate([inv_col_cos, -inv_col_sin], axis=1),
         jnp.concatenate([inv_col_sin, inv_col_cos], axis=1)], axis=0)
    cols_back = _E("wd,noad->noaw", inv_col_block,
                   jnp.concatenate([mixed_re, mixed_im], axis=-1))
    cols_re, cols_im = cols_back[..., :n_cols], cols_back[..., n_cols:]
    row_band, _ = row_band_of(n_modes, n_rows)
    inv_row_cos, inv_row_sin = _dft_rows(row_band, n_rows, +1)
    inv_row_block = jnp.concatenate([inv_row_cos, -inv_row_sin], axis=0)
    field = _E("ha,noaw->nohw", inv_row_block.T,
               jnp.concatenate([cols_re, cols_im], axis=2))
    return (field / (n_rows * n_cols)).astype(F)


def build_variants(n_modes, n_rows, n_cols, n_ch, fused_ok):
    """variant name -> fn(weight, x) for the whole conv, plus the two half-pipelines."""
    def full(wt, x):
        re, im = forward_modes(x, n_modes)
        return inverse_field(gemm_blocked(wt["w_re"], wt["w_im"], re, im),
                             n_modes, n_rows, n_cols, n_ch)

    def fft_only(wt, x):
        re, im = forward_modes(x, n_modes)
        return inverse_field(jnp.concatenate([re, im], axis=1),
                             n_modes, n_rows, n_cols, n_ch)

    def gemm_only(wt, x):
        re, im = forward_modes(jax.lax.stop_gradient(x), n_modes)
        return gemm_blocked(wt["w_re"], wt["w_im"], re, im)

    def grouped(wt, x):
        re, im = forward_modes(x, n_modes)
        return inverse_field(gemm_grouped(wt["w_re"], wt["w_im"], re, im, 2),
                             n_modes, n_rows, n_cols, n_ch)

    def fused(wt, x):
        re, im = forward_modes_fused(x, n_modes)
        return inverse_field(gemm_blocked(wt["w_re"], wt["w_im"], re, im),
                             n_modes, n_rows, n_cols, n_ch)

    def stackcols(wt, x):
        re, im = forward_modes_stackcols(x, n_modes)
        return inverse_field(gemm_blocked(wt["w_re"], wt["w_im"], re, im),
                             n_modes, n_rows, n_cols, n_ch)

    out = {"full": full, "fft_only": fft_only, "gemm_only": gemm_only, "grouped": grouped,
           "stackcols": stackcols}
    if fused_ok:
        out["fused2d"] = fused
    only = os.environ.get("DOCS_FILL_ONLY")
    if only:
        keep = only.split(",")
        out = {k: v for k, v in out.items() if k in keep}
    return out


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision = highest", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, n_ch, n_modes, grid in CASES:
        half, band = n_modes // 2, n_modes
        entries = band * half * grid * grid
        fused_ok = entries <= FUSED_MAX_ENTRIES
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        scale = 1.0 / (n_ch * n_ch)
        weight = {"w_re": jax.random.normal(k1, (n_ch, n_ch, band, half), F) * scale,
                  "w_im": jax.random.normal(k2, (n_ch, n_ch, band, half), F) * scale}
        variants = build_variants(n_modes, grid, grid, n_ch, fused_ok)

        print(f"\n=== {label}: C={n_ch}, n_modes={n_modes}, grid {grid}x{grid}"
              f"{'' if fused_ok else '  (fused2d skipped: matrix too large)'} ===", flush=True)

        # agreement against the shipping conv, on the variants that compute the whole thing
        xs = jax.random.normal(jax.random.PRNGKey(1), (4, n_ch, grid, grid), F)
        ref = np.asarray(spectral_conv(weight, xs, n_modes))
        den = float(np.sqrt(np.sum(ref ** 2)))
        for name in ("full", "grouped", "fused2d", "stackcols"):
            if name not in variants:
                continue
            got = np.asarray(jax.jit(variants[name])(weight, xs))
            err = float(np.sqrt(np.sum((got - ref) ** 2))) / den
            print(f"  rel err vs spectral_conv  {name:<10}{err:.3e}", flush=True)

        print(f"  {'variant':<11}{'B':>6}{'wall us/sample':>16}", flush=True)
        for name, fn in variants.items():
            for B in BATCHES:
                tag = f"{label}_{name}_{B}"
                try:
                    f = jax.jit(fn)
                    x = jax.random.normal(jax.random.PRNGKey(2), (B, n_ch, grid, grid), F)
                    jax.block_until_ready(f(weight, x))
                    ts = []
                    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                        for _ in range(REPS):
                            t0 = time.perf_counter()
                            jax.block_until_ready(f(weight, x))
                            ts.append(time.perf_counter() - t0)
                    us = sorted(ts)[REPS // 2] * 1e6 / B
                    print(f"  {name:<11}{B:>6}{us:>16.2f}", flush=True)
                    rows.append({"case": label, "variant": name, "batch": B,
                                 "us_per_sample": us, "status": "ok", "trace": tag})
                except Exception as exc:
                    why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    print(f"  {name:<11}{B:>6}{why:>16}", flush=True)
                    rows.append({"case": label, "variant": name, "batch": B,
                                 "us_per_sample": float("nan"), "status": why, "trace": ""})
                save(rows)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/fill.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("\nFILL_DONE", flush=True)


if __name__ == "__main__":
    main()
