"""Full-layer inference over method x batch x dtype, reporting H1 loss and device time."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import data, layers, model
from ...fno.layers import h1_rel
from ...fno.spectral import _E, _dft_rows

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
W = os.environ.get("DOCS_W", "tpu/results/h1/h1w")
CSV = os.environ.get("DOCS_MX_CSV", "/tmp/docs_matrix.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_matrix")
REPS = 15
FIELDS = ["case", "grid", "channels", "n_modes", "method", "dtype", "batch",
          "us_per_sample", "h1", "rel_l2", "status", "trace"]

# (label, loader, grid, channels, n_modes, batches to try)
CASES = [
    ("Burgers",  lambda: data.load_burgers(ROOT),   16, 24, 16, (8, 32, 128, 256)),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 32, 24, 16, (8, 32, 128, 256)),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 64, 24, 16, (8, 32, 128, 256)),
    ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=16, n_test=1000),
     128, 64, 64, (4, 8, 16, 32)),
]
WANT_M = [m.strip() for m in os.environ.get(
    "DOCS_METHODS", "reim,coupled,reduced").split(",")]


def band_of(n_modes, n_rows):
    half = n_modes // 2
    return np.concatenate([np.arange(half), np.arange(n_rows - half, n_rows)]), half


# ----------------------------------------------------------- row passes --

def _band_closure(band, period):
    """The band closed under negation, with the index maps back onto it.

    Separating a conjugate-symmetric pair at wavenumber k reads the spectrum at k AND at -k, so
    both have to be present in whatever set gets computed.  This returns that set, together with
    where each k and each -k sits inside it.

    Returns (needed, at_k, at_minus_k), where `needed` is the sorted row set to build and the two
    index arrays are len(band) long.
    """
    needed = np.unique(np.concatenate([band % period, (-band) % period]))
    position = {int(row): i for i, row in enumerate(needed)}

    at_k = jnp.asarray([position[int(k % period)] for k in band])
    at_minus_k = jnp.asarray([position[int((-k) % period)] for k in band])
    return needed, at_k, at_minus_k


def rows_for(x, row_idx, dtype):
    """The row DFT evaluated at exactly `row_idx` -- the one row pass, parameterised by rows.

    Nothing outside `row_idx` is computed, so there is no larger result to slice.  cos and sin
    hit the same operand, so they go as one matmul with the two matrices stacked along the
    output axis.

    x (n, c, h, w) -> (re, im), each (n, c, len(row_idx), w).
    """
    n_rows = x.shape[-2]
    cos, sin = (m.astype(dtype) for m in _dft_rows(row_idx, n_rows, -1))
    stacked = jnp.concatenate([cos, sin], axis=0)

    both = _E("ah,nchw->ncaw", stacked, x.astype(dtype))
    n_keep = len(row_idx)
    return both[:, :, :n_keep], both[:, :, n_keep:]


def rows_reim(x, n_modes, dtype):
    """The real/imag row DFT, evaluated at the kept band.

    There is one row method.  `n_modes` chooses the row set: the band for a truncated transform,
    every row when n_modes = N.  A "partial" transform is just this with n_modes fixed below N,
    not a separate algorithm.
    """
    band, _ = band_of(n_modes, x.shape[-2])
    return rows_for(x, band, dtype)


def rows_coupled(x, n_modes, dtype):
    """Two real channels carried as one complex signal, then separated again.

    The first half of the channels goes in the real part and the second half in the imaginary
    part, so a single complex transform does the work of two real ones.  Separating them
    afterwards reads the spectrum at k and at -k, so the rows built are the band closed under
    negation -- about n_modes of them, never the whole spectrum.
    """
    n_rows = x.shape[-2]
    n_channels = x.shape[1]
    band, _ = band_of(n_modes, n_rows)
    needed, at_k, at_minus_k = _band_closure(band, n_rows)

    cos, sin = (m.astype(dtype) for m in _dft_rows(needed, n_rows, -1))
    first = x[:, :n_channels // 2].astype(dtype)
    second = x[:, n_channels // 2:].astype(dtype)

    # (cos + i sin) @ (first + i second) as three real matmuls instead of four.
    transform = lambda matrix, values: _E("ah,nchw->ncaw", matrix, values)
    cos_first = transform(cos, first)
    sin_second = transform(sin, second)
    cross = transform((cos + sin).astype(dtype), (first + second).astype(dtype))

    packed_re = cos_first - sin_second
    packed_im = cross - cos_first - sin_second

    at_re = jnp.take(packed_re, at_k, axis=2)
    at_im = jnp.take(packed_im, at_k, axis=2)
    mirror_re = jnp.take(packed_re, at_minus_k, axis=2)
    mirror_im = jnp.take(packed_im, at_minus_k, axis=2)

    # A[k] = (Z[k] + conj(Z[-k])) / 2 and B[k] = (Z[k] - conj(Z[-k])) / 2i.
    first_re = 0.5 * (at_re + mirror_re)
    first_im = 0.5 * (at_im - mirror_im)
    second_re = 0.5 * (at_im + mirror_im)
    second_im = -0.5 * (at_re - mirror_re)

    return (jnp.concatenate([first_re, second_re], axis=1),
            jnp.concatenate([first_im, second_im], axis=1))


def rows_reduced(x, n_modes, dtype):
    """Even and odd samples as one complex signal of half the length, twisted back to full.

    even[p] = x[2p] and odd[p] = x[2p+1], so a matrix indexed by the FULL row index n with
    p = n // 2 reaches both streams without materialising either: the even/odd split becomes the
    matmul's read pattern rather than a pair of strided copies.  Only the half-length frequencies
    the band maps onto are built.
    """
    n_rows = x.shape[-2]
    half_length = n_rows // 2
    band, _ = band_of(n_modes, n_rows)
    needed, at_k, at_minus_k = _band_closure(band, half_length)

    sample = np.arange(n_rows)
    is_even = sample % 2 == 0
    angle = (-2.0 * np.pi / half_length) * np.outer(needed, sample // 2)
    cos, sin = np.cos(angle), np.sin(angle)

    # One matrix over the full row index: even columns carry the even stream, odd the odd one.
    real_rows = np.where(is_even[None, :], cos, -sin)
    imag_rows = np.where(is_even[None, :], sin, cos)
    stacked = jnp.asarray(np.concatenate([real_rows, imag_rows], axis=0)).astype(dtype)

    both = _E("ah,nchw->ncaw", stacked, x.astype(dtype))
    n_needed = needed.shape[0]
    packed_re = both[:, :, :n_needed]
    packed_im = both[:, :, n_needed:]

    at_re = jnp.take(packed_re, at_k, axis=2)
    at_im = jnp.take(packed_im, at_k, axis=2)
    mirror_re = jnp.take(packed_re, at_minus_k, axis=2)
    mirror_im = -jnp.take(packed_im, at_minus_k, axis=2)

    even_re = 0.5 * (at_re + mirror_re)
    even_im = 0.5 * (at_im + mirror_im)
    odd_re = 0.5 * (at_im - mirror_im)
    odd_im = -0.5 * (at_re - mirror_re)

    # Twist the odd half by exp(-2 pi i k / n_rows), then add the halves back together.
    twist = (-2.0 * jnp.pi / n_rows) * jnp.asarray(band).astype(F)
    twist_re = jnp.cos(twist)[None, None, :, None]
    twist_im = jnp.sin(twist)[None, None, :, None]

    rows_re = even_re + (twist_re * odd_re - twist_im * odd_im)
    rows_im = even_im + (twist_re * odd_im + twist_im * odd_re)
    return rows_re, rows_im


ROWS = {
    "reim": rows_reim,
    "coupled": rows_coupled,
    "reduced": rows_reduced,
}


def make_conv(method):
    """A spectral conv whose row pass is `method`; columns, GEMM and inverse are shared."""
    rows_fn = ROWS[method]

    def conv(weight, x, n_modes, dtype=F):
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        band, half = band_of(n_modes, n_rows)
        n_ch = (weight["w_block"].shape[3] // 2 if "w_block" in weight
                else weight["w_re"].shape[3])
        rows_re, rows_im = rows_fn(x, n_modes, dtype)

        cos, sin = (c.astype(dtype) for c in _dft_rows(np.arange(half), n_cols, -1))
        col = jnp.concatenate([cos, sin], axis=0)
        over = lambda z: _E("dw,ncaw->ncad", col, z)
        f_re, f_im = over(rows_re), over(rows_im)
        modes_re = f_re[..., :half] - f_im[..., half:]
        modes_im = f_re[..., half:] + f_im[..., :half]

        wb = weight.get("w_block")
        if wb is None:
            wb = layers.spectral_block(weight["w_re"], weight["w_im"], dtype)
        elif wb.dtype != dtype:
            wb = wb.astype(dtype)
        mixed = _E("adio,niad->noad", wb, jnp.concatenate([modes_re, modes_im], axis=1))
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
    return conv


def forward_with(conv, P, x, n_modes, dtype):
    x = layers.add_grid(x)
    x = layers.channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        x = layers.gelu(conv(blk["filter"], x, n_modes, dtype) + layers.conv1x1(blk["skip"], x))
    return layers.channel_mlp(P["proj"], x)


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("GEMM is blocked + ADIO + prebuilt for every method\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for label, loader, N, C, m, batches in CASES:
        try:
            _, _, xte, yte = loader()
        except Exception as exc:
            print(f"{label}: data skipped ({exc})", flush=True)
            continue
        path = os.path.join(W, f"{label}_h1.npz")
        if not os.path.exists(path):
            print(f"{label}: weights missing at {path}", flush=True)
            continue
        P = model.load_params(path, channels=C, n_modes=m, n_layers=4)
        print(f"--- {label}: grid {N}x{N}, C={C}, n_modes={m}", flush=True)
        for dname, dt in (("f32", F), ("bf16", BF)):
            Pb = layers.prebuild_blocks(P, dt)
            for method in WANT_M:
                conv = make_conv(method)
                for B in batches:
                    n = min((len(xte) // B) * B, 256)
                    if n == 0:
                        continue
                    tag = f"{label}_{method}_{dname}_{B}"
                    try:
                        f = jax.jit(lambda z, _c=conv, _p=Pb, _d=dt:
                                    forward_with(_c, _p, z, m, _d))
                        x = jnp.asarray(np.asarray(xte[:B], np.float32))
                        jax.block_until_ready(f(x))
                        h1s, l2s, cnt = 0.0, 0.0, 0
                        for s in range(0, n, B):
                            xb = jnp.asarray(np.asarray(xte[s:s + B], np.float32))
                            yb = jnp.asarray(np.asarray(yte[s:s + B], np.float32))
                            p = f(xb)
                            h1s += float(h1_rel(p, yb)); cnt += p.shape[0]
                            ax = tuple(range(1, yb.ndim))
                            l2s += float(jnp.sum(jnp.sqrt(jnp.sum((p - yb) ** 2, axis=ax))
                                                 / jnp.sqrt(jnp.sum(yb ** 2, axis=ax))))
                        h1, l2 = h1s / cnt, l2s / cnt
                        ts = []
                        with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                            for _ in range(REPS):
                                t0 = time.perf_counter()
                                jax.block_until_ready(f(x))
                                ts.append(time.perf_counter() - t0)
                        us = sorted(ts)[REPS // 2] * 1e6 / B
                        print(f"    {method:<9}{dname:<6}B={B:<5}{us:>9.2f} us  "
                              f"H1 {h1:.4e}  L2 {l2:.4e}", flush=True)
                        rows.append({"case": label, "grid": N, "channels": C, "n_modes": m,
                                     "method": method, "dtype": dname, "batch": B,
                                     "us_per_sample": us, "h1": h1, "rel_l2": l2,
                                     "status": "ok", "trace": tag})
                    except Exception as exc:
                        why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                        print(f"    {method:<9}{dname:<6}B={B:<5}{why:>9}", flush=True)
                        rows.append({"case": label, "grid": N, "channels": C, "n_modes": m,
                                     "method": method, "dtype": dname, "batch": B,
                                     "us_per_sample": float("nan"), "h1": float("nan"),
                                     "rel_l2": float("nan"), "status": why, "trace": ""})
                    save(rows)
        print(flush=True)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/matrix.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)}")
    print("MATRIX_DONE", flush=True)


if __name__ == "__main__":
    main()
