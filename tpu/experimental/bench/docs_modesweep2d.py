"""2D row-pass methods over the full n_modes x N grid, each at its own best batch.

The question this answers: mode truncation is what makes the banded matrix cheap, so as
n_modes approaches N the band stops being a saving and every method converges on the same
full transform.  Sweeping n_modes to N shows whether the ranking survives that limit.

The grid stops at N = 1024, which is the largest size where the n_modes = N diagonal still
fits: the channel weight is 8*C^2*n_modes^2 bytes, quadratic in both, so n_modes = 2048 alone
would need 19 GB.  Beyond 1024 the diagonal cannot be reached at all and the off-diagonal cells
repeat what the smaller grids already show.
"""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno.spectral import _E, _dft_rows
from .docs_matrix import ROWS, band_of

CH = int(os.environ.get("SW_CH", "24"))
NS = [int(v) for v in os.environ.get("SW_NS", "16,32,64,128,256,512,1024").split(",")]
MS = [int(v) for v in os.environ.get("SW_MS", "4,8,16,32,64,128,256,512,1024").split(",")]
METHODS = [s.strip() for s in os.environ.get(
    "SW_METHODS", "reim,coupled,reduced").split(",")]
HBM_BUDGET = float(os.environ.get("SW_BUDGET", 9e9))
REPS = int(os.environ.get("SW_REPS", "12"))
CSV = os.environ.get("SW_CSV", "/tmp/modesweep2d.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/prof_modesweep2d")
FIELDS = ["grid", "n_modes", "channels", "method", "batch", "us_per_sample",
          "rel_err", "status", "trace"]


def weight_bytes(n_modes):
    return 8 * CH * CH * n_modes * n_modes


def batches_for(n_modes, n):
    """Powers of two that leave room for the weight plus a handful of activations."""
    room = HBM_BUDGET - weight_bytes(n_modes)
    if room <= 0:
        return []
    if room <= 0:
        return []
    per_sample = CH * n * n * 4 * 6          # ~6 live tensors of field size
    cap = min(128, int(room // per_sample))
    cands = [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= cap]
    return cands[-4:] if len(cands) > 4 else (cands or [1])


def build_conv(method):
    """All methods share columns, GEMM and inverse; only the row pass differs."""
    return _conv_with_rows(ROWS[method])


def _conv_with_rows(rows_fn):
    """make_conv's body with an injected row pass, so columns/GEMM/inverse stay shared."""
    def conv(weight, x, n_modes, dtype=F):
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        band, half = band_of(n_modes, n_rows)
        n_ch = weight["w_block"].shape[3] // 2
        rows_re, rows_im = rows_fn(x, n_modes, dtype)
        cos, sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
        over = lambda z: _E("dw,ncaw->ncad", jnp.concatenate([cos, sin], axis=0), z)
        f_re, f_im = over(rows_re), over(rows_im)
        modes_re = f_re[..., :half] - f_im[..., half:]
        modes_im = f_re[..., half:] + f_im[..., :half]
        mixed = _E("adio,niad->noad", weight["w_block"].astype(dtype),
                   jnp.concatenate([modes_re, modes_im], axis=1))
        m_re, m_im = mixed[:, :n_ch], mixed[:, n_ch:]
        fold = jnp.where(jnp.arange(half) == 0, 1.0, 2.0).astype(F)
        icc, ics = _dft_rows(np.arange(half), n_cols, +1)
        icc, ics = (icc * fold[:, None]).T, (ics * fold[:, None]).T
        inv_col = jnp.concatenate([jnp.concatenate([icc, -ics], axis=1),
                                   jnp.concatenate([ics, icc], axis=1)], axis=0)
        cb = _E("wd,noad->noaw", inv_col, jnp.concatenate([m_re, m_im], axis=-1))
        irc, irs = _dft_rows(band, n_rows, +1)
        field = _E("ha,noaw->nohw", jnp.concatenate([irc, -irs], axis=0).T,
                   jnp.concatenate([cb[..., :n_cols], cb[..., n_cols:]], axis=2))
        return (field / (n_rows * n_cols)).astype(F)
    return conv


def timed(fn, weight, x, tag):
    """Median wall clock plus an xprof trace; device time is read from the trace later."""
    jax.block_until_ready(fn(weight, x))
    samples = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(weight, x))
            samples.append(time.perf_counter() - t0)
    return sorted(samples)[REPS // 2] * 1e6


def save(rows):
    with open(CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"channels={CH}  reps={REPS}  budget={HBM_BUDGET / 1e9:.1f} GB", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for n in NS:
        for n_modes in [m for m in MS if m <= n]:
            batches = batches_for(n_modes, n)
            if not batches:
                print(f"N={n:<5} m={n_modes:<5} SKIP  w_block "
                      f"{weight_bytes(n_modes) / 1e9:.1f} GB exceeds budget", flush=True)
                continue
            # One allocation, not three: building w_re and w_im and then concatenating held
            # ~2.5x the block live at once.  Only the shape drives the timing, and every
            # method shares this tensor, so the cross-method agreement check still holds.
            scale = 1.0 / (CH * CH)
            weight = {"w_block": scale * jax.random.normal(
                jax.random.PRNGKey(0), (n_modes, n_modes // 2, 2 * CH, 2 * CH), F)}
            print(f"N={n:<5} m={n_modes:<5} w_block "
                  f"{weight_bytes(n_modes) / 1e6:>8.1f} MB  batches {batches}", flush=True)
            reference = None
            for method in METHODS:
                conv = build_conv(method)
                best = None
                for batch in batches:
                    tag = f"N{n}_m{n_modes}_{method}_B{batch}"
                    try:
                        x = jax.random.normal(jax.random.PRNGKey(3), (batch, CH, n, n), F)
                        # weight goes in as an ARGUMENT, not a closure capture: captured
                        # arrays become compile-time constants and XLA tries to fold them,
                        # which at multi-GB block sizes never finishes compiling.
                        fn = jax.jit(lambda w, z, _c=conv: _c(w, z, n_modes, F))
                        out = jax.block_until_ready(fn(weight, x))
                        got = np.asarray(out[:1, :2])
                        if method == METHODS[0] and batch == batches[0]:
                            reference, err = got, 0.0
                        elif reference is not None and got.shape[0] >= reference.shape[0]:
                            k = reference.shape[0]
                            err = float(np.linalg.norm(got[:k] - reference)
                                        / np.linalg.norm(reference))
                        else:
                            err = float("nan")
                        us = timed(fn, weight, x, tag) / batch
                        rows.append({"grid": n, "n_modes": n_modes, "channels": CH,
                                     "method": method, "batch": batch,
                                     "us_per_sample": us, "rel_err": err,
                                     "status": "ok", "trace": tag})
                        if best is None or us < best[0]:
                            best = (us, batch, err)
                        del x, out, got
                    except Exception as exc:
                        why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                        rows.append({"grid": n, "n_modes": n_modes, "channels": CH,
                                     "method": method, "batch": batch,
                                     "us_per_sample": float("nan"), "rel_err": float("nan"),
                                     "status": why, "trace": ""})
                    save(rows)
                if best is None:
                    print(f"    {method:<9} all batches failed", flush=True)
                else:
                    print(f"    {method:<9}{best[0]:>10.3f} us/sample  @B={best[1]:<5}"
                          f" err {best[2]:.2e}", flush=True)
            del weight, reference
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/modesweep2d.tar.gz "
              f"{os.path.basename(LOGDIR)} {os.path.basename(CSV)}")
    print("MODESWEEP2D_DONE", flush=True)


if __name__ == "__main__":
    main()
