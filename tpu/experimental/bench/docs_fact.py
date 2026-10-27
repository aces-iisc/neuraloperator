"""F-FNO and TFNO against the dense filter, on the plain jnp.fft baseline transform.

Every variant runs the SAME transform -- jnp.fft.rfft2, truncate, irfft2, complex64, the
standard neuralop path -- so the only thing that differs between them is the channel filter.
None of the transform work from Sections 4-5 is in this path, deliberately: a claim about
factorising the filter has to be measured against the baseline, not against our transform.

Time is swept over N on synthetic inputs; accuracy needs real targets, so it is trained on the
Darcy splits that exist at 32 and 64.
"""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ..fno import factorized as fz
from ..fno import factorized_opt as fzo
from .parse_ops import bucket_times

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
# the four shipped datasets, with the channels and band each one is configured with
ONLY = os.environ.get("FACT_ONLY", "")
CASES = [("Burgers",  lambda: _data().load_burgers(ROOT),                      24, 16, 32),
         ("Darcy-32", lambda: _data().load_darcy(ROOT, 32),                    24, 16, 32),
         ("Darcy-64", lambda: _data().load_darcy(ROOT, 64),                    24, 16, 32),
         ("NS-128",   lambda: _data().load_ns(ROOT, 128, n_train=16, n_test=32),  64, 64, 4)]


def _data():
    from ...fno import data
    return data
CSV = os.environ.get("DOCS_FACT_CSV", "/tmp/docs_fact.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/prof_fact")
REPS = 15
RANK = int(os.environ.get("FACT_R", "8"))
STEPS = int(os.environ.get("FACT_STEPS", "60"))
# FACT_BATCH="Burgers=512,Darcy-32=128" overrides the per-dataset batch in CASES.
BATCH_OVERRIDE = dict(
    (k, int(v)) for k, v in
    (pair.split("=") for pair in os.environ.get("FACT_BATCH", "").split(",") if pair))
FIELDS = ["engine", "transform", "filter_form", "case", "grid", "variant", "batch",
          "params", "us_per_sample", "device_us_per_sample", "mxu_us", "vpu_us",
          "copy_us", "dma_us", "rel_l2", "status", "trace"]
BUCKETS = ["MXU matmul", "VPU / fusion", "reshape / copy", "DMA / HBM"]

VARIANTS = [("dense", fz.dense_init, fz.dense_filter, fzo.dense_filter_opt, fzo.dense_block, {}),
            ("ffno", fz.sep_init, fz.sep_filter, fzo.sep_filter_opt, fzo.sep_block, {}),
            ("tfno", fz.cp_init, fz.cp_filter, fzo.cp_filter_opt, fzo.cp_block, {"rank": RANK})]


def prebuild(P, to_block):
    """Hoist every block out of the per-call path, the way layers.prebuild_blocks does."""
    return {**P, "blocks": [{**b, "filter": to_block(b["filter"])} for b in P["blocks"]]}


def forward_jnp_filter(P, x, filt, m):
    """The baseline spectral conv with the channel mix swapped out.

    jnp.fft.rfft2 -> keep the low m x m corner -> `filt` -> irfft2.  Identical to
    factorized.forward_jnp except that the filter is a parameter rather than hardcoded.
    """
    x = fz.add_grid(x)
    x = fz.channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        batch, _, n_rows, n_cols = x.shape
        spectrum = jnp.fft.rfft2(x, axes=(-2, -1))
        low = spectrum[..., :m, :m]
        mixed_re, mixed_im = filt(blk["filter"], low.real, low.imag)
        mixed = (mixed_re + 1j * mixed_im).astype(spectrum.dtype)
        out_ch = mixed_re.shape[1]
        full = jnp.zeros((batch, out_ch, n_rows, n_cols // 2 + 1),
                         spectrum.dtype).at[..., :m, :m].set(mixed)
        field = jnp.fft.irfft2(full, s=(n_rows, n_cols), axes=(-2, -1))
        x = fz.gelu(field + fz.conv1x1(blk["skip"], x))
    return fz.channel_mlp(P["proj"], x)


def jnp_arm(P, x, filt, m):
    return forward_jnp_filter(P, x, filt, m)


def ours_arm(P, x, filt, m):
    return fz.forward(P, x, filt, m)


def timed(fn, x, tag):
    """Median wall time and the device-time split, both per call, from one trace."""
    path = os.path.join(LOGDIR, tag)
    jax.block_until_ready(fn(x))
    samples = []
    with jax.profiler.trace(path):
        for _ in range(REPS):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(x))
            samples.append(time.perf_counter() - t0)
    return sorted(samples)[REPS // 2] * 1e6, bucket_times(path, REPS)


def save(rows):
    with open(CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("baseline transform = jnp.fft.rfft2 / irfft2, complex64\n", flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    # Three engines.  "baseline" is the plain jnp.fft path a published FNO uses, with the filters
    # in their published form too.  "baseline+opt" keeps that transform and swaps in the blocked,
    # stacked, hoisted filter of Section 5, which isolates what the factorisation is worth when
    # none of the transform work is helping.  "ours" has both.
    # The full 2x2: the transform is either jnp.fft or ours, and the channel filter is either
    # the form its paper issues -- separate real/imaginary einsums with the mode axes trailing --
    # or the blocked ADIO form of Section 5, one real matmul against a stacked weight.  Holding
    # one axis fixed and moving the other is what separates the two contributions.
    engines = [("native+native", jnp_arm, False),
               ("native+ours", jnp_arm, True),
               ("ours+native", ours_arm, False),
               ("ours+ours", ours_arm, True)]
    only = [e.strip() for e in os.environ.get("FACT_ENGINES", "").split(",") if e.strip()]
    if only:
        engines = [e for e in engines if e[0] in only]
    print(f"  {'engine':<14}{'case':<10}{'variant':<8}{'params':>11}"
          f"{'device us':>12}{'wall us':>12}{'rel L2':>12}", flush=True)
    for label, loader, channels, modes, batch in CASES:
        if ONLY and label != ONLY:
            continue
        try:
            xtr, ytr, xte, yte = loader()
        except Exception as exc:
            print(f"  {label:<10}SKIPPED  {type(exc).__name__}", flush=True)
            continue
        grid = xte.shape[-1]
        batch = BATCH_OVERRIDE.get(label, batch)
        x = jnp.asarray(np.asarray(xte[:batch], np.float32))
        for eng_name, eng, blocked in engines:
          for name, init, filt, filt_opt, to_block, kw in VARIANTS:
            kw = dict(kw)
            tag = f"{eng_name}_{label}_{name}"
            try:
                P = fz.build(jax.random.PRNGKey(0), init, 1, 1, channels, modes, 4, **kw)
                use = filt
                if blocked:                       # blocked + stacked + ADIO, hoisted
                    P, use = prebuild(P, to_block), filt_opt
                fwd = lambda P_, z, _f=use, _m=modes, _e=eng: _e(P_, z, _f, _m)
                fn = jax.jit(lambda z, _P=P, _f=fwd: _f(_P, z))
                us, buckets = timed(fn, x, tag)
                us /= x.shape[0]
                per = lambda key: buckets.get(key, 0.0) / x.shape[0]
                err = fz.quick_train(fwd, P, xtr, ytr, xte, yte, steps=STEPS,
                                     batch=min(32, xtr.shape[0]))
                xform, form = eng_name.split("+")
                rows.append({"engine": eng_name, "transform": xform, "filter_form": form,
                             "case": label, "grid": grid, "variant": name,
                             "batch": x.shape[0], "params": fz.n_params(P),
                             "us_per_sample": us, "device_us_per_sample": per("total"),
                             "mxu_us": per(BUCKETS[0]), "vpu_us": per(BUCKETS[1]),
                             "copy_us": per(BUCKETS[2]), "dma_us": per(BUCKETS[3]),
                             "rel_l2": err, "status": "ok", "trace": tag})
                print(f"  {eng_name:<14}{label:<10}{name:<8}{fz.n_params(P):>11,}"
                      f"{per('total'):>12.2f}{us:>12.2f}{err:>12.4e}", flush=True)
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                xform, form = eng_name.split("+")
                rows.append({"engine": eng_name, "transform": xform, "filter_form": form,
                             "case": label, "grid": grid, "variant": name,
                             "batch": x.shape[0], "params": 0, "us_per_sample": float("nan"),
                             "device_us_per_sample": float("nan"), "mxu_us": float("nan"),
                             "vpu_us": float("nan"), "copy_us": float("nan"),
                             "dma_us": float("nan"), "rel_l2": float("nan"),
                             "status": why, "trace": ""})
                print(f"  {eng_name:<14}{label:<10}{name:<8}{why:>19}  "
                      f"{str(exc)[:70]}", flush=True)
            save(rows)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/fact.tar.gz {os.path.basename(LOGDIR)} "
              f"{os.path.basename(CSV)} 2>/dev/null")
    print("\nFACT_DONE", flush=True)


if __name__ == "__main__":
    main()
