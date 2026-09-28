"""Inference at the repository's configured test batch, and at each config's tuned batch."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import data, model
from ..fno.reference import forward as jnp_fwd

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
W = os.environ.get("DOCS_W", "/tmp/docs_full")
CSV = os.environ.get("DOCS_CB_CSV", "/tmp/docs_cfgbatch.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_cb")
FIELDS = ["case", "config", "batch", "batch_kind", "us_per_sample", "rel_l2", "n_eval",
          "status", "trace"]
REPS = 20

# (label, loader, channels, n_modes, configured test batch, B* f32, B* bf16)
CASES = [("Burgers",  lambda: data.load_burgers(ROOT),   24, 16, 16, 128, 128),
         ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 24, 16, 16, 128, 128),
         ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 24, 16, 16, 128, 128),
         ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=16, n_test=1000), 64, 64, 8, 8, 4)]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def rel_l2(f, xte, yte, B, n):
    e = d = 0.0
    for s in range(0, n, B):
        p = np.asarray(f(jnp.asarray(np.asarray(xte[s:s+B], np.float32))))
        g = np.asarray(yte[s:s+B], np.float32)
        ax = tuple(range(1, g.ndim))
        e += float(np.sum(np.sqrt(np.sum((p-g)**2, axis=ax))))
        d += float(np.sum(np.sqrt(np.sum(g**2, axis=ax))))
    return e/d


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    print(f"  {'case':<10}{'config':<22}{'B':>5}{'kind':<10}{'us/sample':>11}{'rel L2':>12}", flush=True)
    for label, loader, C, nm, Bcfg, Bf32, Bbf in CASES:
        try:
            _, _, xte, yte = loader()
        except Exception as exc:
            print(f"  {label}: data skipped"); continue
        load = lambda f: model.load_params(os.path.join(W, f"{label}_{f}.npz"),
                                           channels=C, n_modes=nm, n_layers=4)
        setups = [
            ("baseline jnp.fft", load("jnp_baseline"), jnp_fwd, F, Bcfg, "config"),
            ("ours f32",  load("ours_f32"),  model.forward, F,  Bcfg, "config"),
            ("ours bf16", load("ours_bf16"), model.forward, BF, Bcfg, "config"),
            ("ours f32",  load("ours_f32"),  model.forward, F,  Bf32, "tuned"),
            ("ours bf16", load("ours_bf16"), model.forward, BF, Bbf,  "tuned"),
        ]
        for cname, P, fwd, dt, B, kind in setups:
            n = (len(xte)//B)*B
            if n == 0: continue
            tag = f"{label}_{cname.replace(' ','_').replace('.','')}_{B}"
            try:
                f = jax.jit(lambda z, _p=P, _f=fwd, _d=dt:
                            _f(_p, z, nm) if _f is jnp_fwd else _f(_p, z, nm, _d))
                x = jnp.asarray(np.asarray(xte[:B], np.float32))
                jax.block_until_ready(f(x))
                ts = []
                with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                    for _ in range(REPS):
                        t0 = time.perf_counter(); jax.block_until_ready(f(x))
                        ts.append(time.perf_counter()-t0)
                us = sorted(ts)[REPS//2]*1e6/B
                rl = rel_l2(f, xte, yte, B, n)
                print(f"  {label:<10}{cname:<22}{B:>5}{kind:<10}{us:>11.2f}{rl:>12.4e}", flush=True)
                rows.append({"case": label, "config": cname, "batch": B, "batch_kind": kind,
                             "us_per_sample": us, "rel_l2": rl, "n_eval": n,
                             "status": "ok", "trace": tag})
            except Exception as exc:
                print(f"  {label:<10}{cname:<22}{B:>5}{kind:<10}{type(exc).__name__}", flush=True)
                rows.append({"case": label, "config": cname, "batch": B, "batch_kind": kind,
                             "us_per_sample": float("nan"), "rel_l2": float("nan"),
                             "n_eval": 0, "status": type(exc).__name__, "trace": ""})
            save(rows)
        print(flush=True)
    save(rows)
    os.system("cd /tmp && tar czf /tmp/cb_prof.tar.gz docs_prof_cb docs_cfgbatch.csv")
    print("CFGBATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
