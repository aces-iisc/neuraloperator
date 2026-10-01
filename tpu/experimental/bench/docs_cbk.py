"""Is the ->cbk stage output helping or hurting? Both variants, same code path, swept over N."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

jax.config.update("jax_default_matmul_precision", "highest")
F, ACC, RADIX = jnp.float32, jnp.float32, 128
K = int(os.environ.get("DOCS_K", "256"))
REPS = 20
NS = (256, 512, 1024, 2048, 4096, 8192, 16384)
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_cbk")
CSV = os.environ.get("DOCS_CBK_CSV", "/tmp/docs_cbk.csv")


def dft_reim(size, dt=F):
    k = jnp.arange(size)
    a = (-2.0 * jnp.pi / size) * (k[:, None] * k[None, :]).astype(F)
    c, s = jnp.cos(a).astype(dt), jnp.sin(a).astype(dt)
    return c, s, (c + s).astype(dt)


def twiddle(stage_len, n_cols, cbk):
    """(radix, cols) for the bck layout, (cols, radix) for cbk -- same factors either way."""
    r = jnp.arange(RADIX)
    c = jnp.arange(n_cols)
    ang = ((-2.0 * jnp.pi / stage_len) *
           ((c[:, None] * r[None, :]) if cbk else (r[:, None] * c[None, :])).astype(F))
    return jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]


def staged(sig, n, cbk, dt=F):
    """One radix-128 sweep then a leaf, with the stage transpose either folded or explicit."""
    spec = 'br,rck->cbk' if cbk else 'br,rck->bck'
    mm = lambda m, x: jnp.einsum(spec, m, x, preferred_element_type=ACC)
    cur_re, cur_im = sig.astype(dt), None
    length, batch, real = n, sig.shape[1], True
    while length > RADIX and length % RADIX == 0:
        n_cols = length // RADIX
        c, s, cps = dft_reim(RADIX, dt)
        blk_re = cur_re.reshape(RADIX, n_cols, batch)
        if real:
            sr, si = mm(c, blk_re), mm(s, blk_re)
            real = False
        else:
            blk_im = cur_im.reshape(RADIX, n_cols, batch)
            a = mm(c, blk_re); b = mm(s, blk_im)
            t = mm(cps, blk_re + blk_im)
            sr, si = a - b, t - a - b
        tc, ts = twiddle(length, n_cols, cbk)
        nr, ni = sr * tc - si * ts, sr * ts + si * tc
        if not cbk:
            nr, ni = nr.transpose(1, 0, 2), ni.transpose(1, 0, 2)
        cur_re = nr.reshape(n_cols, RADIX * batch).astype(dt)
        cur_im = ni.reshape(n_cols, RADIX * batch).astype(dt)
        length, batch = n_cols, RADIX * batch
    c, s, cps = dft_reim(length, dt)
    lm = lambda m, x: jnp.matmul(m, x, preferred_element_type=ACC)
    if real:
        return lm(c, cur_re), lm(s, cur_re)
    a = lm(c, cur_re); b = lm(s, cur_im)
    t = lm(cps, cur_re + cur_im)
    return a - b, t - a - b


def timed(f, x, tag):
    """Wall clock plus an xprof trace, since wall clock alone is dispatch-bound at small N."""
    jax.block_until_ready(f(x))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(f(x))
            ts.append(time.perf_counter() - t0)
    return sorted(ts)[REPS // 2] * 1e6


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"K={K}, one radix-{RADIX} stage per case\n", flush=True)
    print(f"{'N':>7}{'n_cols':>8}{'bck (old)':>12}{'cbk (new)':>12}{'cbk gain':>11}", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    for n in NS:
        x = jnp.asarray(np.random.default_rng(0).standard_normal((n, K)).astype(np.float32))
        try:
            a = timed(jax.jit(lambda z: staged(z, n, False)), x, f"N{n}_bck")
            b = timed(jax.jit(lambda z: staged(z, n, True)), x, f"N{n}_cbk")
            rows.append({"n": n, "n_cols": n // RADIX, "variant": "bck",
                         "wall_us": a, "trace": f"N{n}_bck"})
            rows.append({"n": n, "n_cols": n // RADIX, "variant": "cbk",
                         "wall_us": b, "trace": f"N{n}_cbk"})
            with open(CSV, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["n", "n_cols", "variant", "wall_us", "trace"])
                w.writeheader(); w.writerows(rows)
            r0 = np.asarray(staged(x, n, False)[0])
            r1 = np.asarray(staged(x, n, True)[0])
            agree = float(np.abs(r0 - r1).max())
            print(f"{n:>7}{n // RADIX:>8}{a:>12.1f}{b:>12.1f}{a / b:>10.2f}x"
                  f"   (max|diff| {agree:.1e})", flush=True)
        except Exception as exc:
            print(f"{n:>7}{n // RADIX:>8}  FAILED {str(exc)[:70]}", flush=True)
    os.system(f"cd /tmp && tar czf /tmp/cbk.tar.gz {os.path.basename(LOGDIR)} {os.path.basename(CSV)}")
    print("\nCBK_DONE", flush=True)


if __name__ == "__main__":
    main()
