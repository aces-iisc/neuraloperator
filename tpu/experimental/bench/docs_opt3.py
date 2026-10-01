"""Full inference after the three layout changes, repeated end to end to show the spread."""
import csv, os, statistics, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import data, layers, model
from ..fno.reference import forward as jnp_fwd

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
W = os.environ.get("DOCS_W", "tpu/results/baseline")
CSV = os.environ.get("DOCS_OPT3_CSV", "/tmp/docs_opt3.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_opt3")
RUNS = int(os.environ.get("DOCS_RUNS", "10"))
REPS = 20
FIELDS = ["case", "variant", "batch", "run", "us_per_sample", "rel_l2", "status"]

# (label, loader, channels, n_modes, eval batch)
CASES = [("Burgers", lambda: data.load_burgers(ROOT), 24, 16, 256),
         ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 24, 16, 256),
         ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 24, 16, 128),
         ("NS-128", lambda: data.load_ns(ROOT, 128, n_train=16, n_test=1000), 64, 64, 8)]


def rel_l2(f, xte, yte, batch, n):
    err = den = 0.0
    for s in range(0, n, batch):
        p = np.asarray(f(jnp.asarray(np.asarray(xte[s:s + batch], np.float32))))
        g = np.asarray(yte[s:s + batch], np.float32)
        ax = tuple(range(1, g.ndim))
        err += float(np.sum(np.sqrt(np.sum((p - g) ** 2, axis=ax))))
        den += float(np.sum(np.sqrt(np.sum(g ** 2, axis=ax))))
    return err / den


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = highest, {RUNS} end-to-end runs per configuration\n", flush=True)
    rows = []
    for label, loader, C, nm, B in CASES:
        try:
            _, _, xte, yte = loader()
        except Exception as exc:
            print(f"{label}: data skipped ({exc})", flush=True)
            continue
        n = (len(xte) // B) * B
        load = lambda f: model.load_params(os.path.join(W, f"{label}_{f}.npz"),
                                           channels=C, n_modes=nm, n_layers=4)
        Pb, Po, Pf = load("jnp"), load("ours"), load("ours-bf16")

        variants = [("baseline jnp.fft", Pb, jnp_fwd, F),
                    ("ours f32", layers.prebuild_blocks(Po, F), model.forward, F),
                    ("ours bf16", layers.prebuild_blocks(Pf, BF), model.forward, BF)]
        print(f"--- {label}  (C={C}, n_modes={nm}, B={B}, n_eval={n})", flush=True)
        for name, P, fwd, dt in variants:
            f = jax.jit(lambda z, _p=P, _f=fwd, _d=dt:
                        _f(_p, z, nm) if _f is jnp_fwd else _f(_p, z, nm, _d))
            x = jnp.asarray(np.asarray(xte[:B], np.float32))
            try:
                jax.block_until_ready(f(x))            # compile once, outside the timed runs
                r2 = rel_l2(f, xte, yte, B, n)
                per_run = []
                for r in range(RUNS):
                    ts = []
                    for _ in range(REPS):
                        t0 = time.perf_counter()
                        jax.block_until_ready(f(x))
                        ts.append(time.perf_counter() - t0)
                    us = sorted(ts)[REPS // 2] * 1e6 / B
                    per_run.append(us)
                    rows.append({"case": label, "variant": name, "batch": B, "run": r,
                                 "us_per_sample": us, "rel_l2": r2, "status": "ok"})
                lo, hi = min(per_run), max(per_run)
                avg = statistics.mean(per_run)
                sd = statistics.stdev(per_run) if len(per_run) > 1 else 0.0
                print(f"    {name:<18} avg {avg:8.2f}  min {lo:8.2f}  max {hi:8.2f}  "
                      f"delta {hi - lo:7.2f} ({(hi - lo) / avg * 100:5.1f}%)  sd {sd:6.2f}  "
                      f"relL2 {r2:.4e}", flush=True)
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                print(f"    {name:<18} {why}: {str(exc)[:80]}", flush=True)
                rows.append({"case": label, "variant": name, "batch": B, "run": -1,
                             "us_per_sample": float("nan"), "rel_l2": float("nan"),
                             "status": why})
            save(rows)
        print(flush=True)
    save(rows)
    print("OPT3_DONE", flush=True)


if __name__ == "__main__":
    main()
