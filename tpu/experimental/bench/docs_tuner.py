"""Autotuner on random data: times the whole model to pick the batch. No training.

The two decisions are not made the same way.  The band comes from the spectrum of the targets, so
it is read off the data once and held fixed -- searching it on speed alone would always return the
smallest band, which is fastest and least accurate.  The batch is a pure throughput question, so
it is measured, over the window the rules predict.  Both the sweep and the configuration finally
selected are recorded, since the table reports what the tuner chose, not merely what it tried.
"""
import csv, json, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import layers, model
from ...arch import TPU_V5E, ModelSpec
from ...fno.autotune import AutoTune

neighbours_add = AutoTune.neighbours_add
neighbours_mul = AutoTune.neighbours_mul


def predict_batch(channels, n_rows, n_cols, dtype="float32"):
    """The batch the rules start from, for the shape this bench is sweeping."""
    spec = ModelSpec(grid=max(n_rows, n_cols), channels=channels, n_modes=16)
    return AutoTune(TPU_V5E, spec).batch(dtype)[0]
from ...fno.spectral import spectral_conv

CSV = os.environ.get("DOCS_TUNER_CSV", "/tmp/docs_tuner.csv")
FIELDS = ["case", "role", "target", "dtype", "n_modes", "batch", "us_per_sample",
          "m_predicted", "b_predicted", "tune_s", "status"]
TARGETS = [t.strip() for t in os.environ.get("DOCS_TARGETS", "model,layer").split(",")]
REPS = 12

# (label, channels, grid, the band the energy criterion picks).  The bands are what
# docs_energy reads off the targets, so the two tables cannot disagree.
CASES = [("Burgers", 24, (17, 16), 16),
         ("Darcy-32", 24, (32, 32), 24),
         ("Darcy-64", 24, (64, 64), 24),
         ("NS-128", 64, (128, 128), 24)]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def timed(fn, x):
    j = jax.jit(fn)
    jax.block_until_ready(j(x))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); jax.block_until_ready(j(x)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[REPS//2]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("random inputs, no training: this tunes speed only\n", flush=True)
    rows, wall0 = [], time.perf_counter()
    for label, C, (H, W), m_pred in CASES:
        b_pred = predict_batch(C, H, W)
        ms = [m for m in neighbours_add(m_pred, 8, 2) if m <= min(H, W)]
        bs = neighbours_mul(b_pred, 2, lo=1)
        print(f"### {label}  channels={C} grid={H}x{W}   predicted m={m_pred} B={b_pred}", flush=True)
        print(f"    sweeping m {ms}  x  B {bs}", flush=True)
        t_case = time.perf_counter()
        for dt, dn in ((F, "float32"), (BF, "bfloat16")):
            for target in TARGETS:
                best = (None, None, 1e18)
                for m in ms:
                    w = layers._init_spectral_weight(jax.random.PRNGKey(0), C, C, m)
                    P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, channels=C,
                                        n_modes=m, n_layers=4)
                    for B in bs:
                        try:
                            x = jnp.asarray(np.random.default_rng(0).standard_normal(
                                (B, C, H, W)).astype(np.float32))
                            if target == "model":
                                x = jnp.asarray(np.random.default_rng(0).standard_normal(
                                    (B, 1, H, W)).astype(np.float32))
                                fn = lambda z, _p=P, _m=m, _d=dt: model.forward(_p, z, _m, _d)
                            else:
                                fn = lambda z, _w=w, _m=m, _d=dt: spectral_conv(_w, z, _m, _d)
                            us = timed(fn, x)*1e6/B
                            # the band is fixed by the energy rule, so only batches at m_pred
                            # are candidates for the selection
                            if m == m_pred and us < best[2]: best = (m, B, us)
                            rows.append({"case": label, "role": "sweep", "target": target,
                                         "dtype": dn, "n_modes": m, "batch": B,
                                         "us_per_sample": us, "m_predicted": m_pred,
                                         "b_predicted": b_pred, "tune_s": float("nan"),
                                         "status": "ok"})
                        except Exception as exc:
                            rows.append({"case": label, "role": "sweep", "target": target,
                                         "dtype": dn, "n_modes": m, "batch": B,
                                         "us_per_sample": float("nan"), "m_predicted": m_pred,
                                         "b_predicted": b_pred, "tune_s": float("nan"),
                                         "status": type(exc).__name__})
                        save(rows)
                rows.append({"case": label, "role": "selected", "target": target,
                             "dtype": dn, "n_modes": best[0], "batch": best[1],
                             "us_per_sample": best[2], "m_predicted": m_pred,
                             "b_predicted": b_pred,
                             "tune_s": time.perf_counter() - t_case, "status": "ok"})
                print(f"    {target:<6} {dn:<9} selected m={best[0]:<4} B={best[1]:<5} "
                      f"{best[2]:8.2f} us/sample   (rules said m={m_pred} B={b_pred})",
                      flush=True)
        print(f"    tuned in {time.perf_counter()-t_case:.1f} s\n", flush=True)
    save(rows)
    print(f"TUNER_DONE  total {time.perf_counter()-wall0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
