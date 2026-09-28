"""Full training: jnp baseline against our engine in f32 and bf16, at each one's tuned batch."""
import csv, json, os, subprocess, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import data, model, train
from ...fno.layers import n_params
from ..fno.reference import forward as jnp_forward

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
OUT = os.environ.get("DOCS_FULL_OUT", "/tmp/docs_full")
CSV = os.environ.get("DOCS_FULL_CSV", "/tmp/docs_full.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_full")
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "60"))
FIELDS = ["case", "config", "n_modes", "channels", "train_batch", "infer_batch",
          "rel_l2", "train_s", "infer_us", "params", "status", "trace"]

# (label, loader, channels, n_modes, train batch, B* f32, B* bf16)
ALL = {
  "Burgers":  (lambda: data.load_burgers(ROOT),          24, 16, 16, 128, 128),
  "Darcy-32": (lambda: data.load_darcy(ROOT, 32),        24, 16, 8,  128, 128),
  "Darcy-64": (lambda: data.load_darcy(ROOT, 64),        24, 16, 8,  128, 128),
  "NS-128":   (lambda: data.load_ns(ROOT, 128, n_train=10000, n_test=1000), 64, 64, 8, 8, 4),
  "Darcy-421":(lambda: data.load_darcy(ROOT, 421),       24, 16, 8,  8,   8),
}
WANT = [c.strip() for c in os.environ.get("DOCS_CASES", "Burgers,Darcy-32,Darcy-64,NS-128").split(",")]


def save(r):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(r)


def infer_us(P, xte, nm, fwd, dt, B, tag):
    """Wall clock per sample, with a trace captured so device time can be parsed off-box."""
    x = jnp.asarray(np.asarray(xte[:B], np.float32))
    f = jax.jit(lambda z: fwd(P, z, nm) if fwd is jnp_forward else fwd(P, z, nm, dt))
    jax.block_until_ready(f(x))
    ts = []
    with jax.profiler.trace(os.path.join(LOGDIR, tag)):
        for _ in range(20):
            t0 = time.perf_counter(); jax.block_until_ready(f(x)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[10]*1e6/B


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"epochs={EPOCHS}  cases={WANT}", flush=True)
    os.makedirs(OUT, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    print(f"\n  {'case':<11}{'config':<16}{'B tr':>5}{'B inf':>6}{'rel L2':>12}{'train s':>10}{'us/sample':>11}", flush=True)
    for label in WANT:
        loader, C, nm, Btr, Bf32, Bbf = ALL[label]
        try:
            xtr, ytr, xte, yte = loader()
        except Exception as exc:
            print(f"  {label}: data skipped ({type(exc).__name__}: {str(exc)[:50]})", flush=True); continue
        for cname, fwd, dt, Binf in (("jnp baseline", jnp_forward, F, 128),
                                     ("ours f32", model.forward, F, Bf32),
                                     ("ours bf16", model.forward, BF, Bbf)):
            try:
                f = fwd if fwd is jnp_forward else (lambda P, x, m, _d=dt: model.forward(P, x, m, _d))
                t0 = time.perf_counter()
                P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=nm, channels=C, n_layers=4,
                                           steps=EPOCHS, batch=Btr, fwd=f, verbose=False,
                                           save=os.path.join(OUT, f"{label}_{cname.replace(' ','_')}.npz"))
                tag = f"{label}_{cname.replace(' ','_')}_{Binf}"
                us = infer_us(P, xte, nm, fwd, dt, Binf, tag)
                print(f"  {label:<11}{cname:<16}{Btr:>5}{Binf:>6}{rel:>12.4e}{secs:>10.1f}{us:>11.2f}", flush=True)
                rows.append({"case": label, "config": cname, "n_modes": nm, "channels": C,
                             "train_batch": Btr, "infer_batch": Binf, "rel_l2": rel,
                             "train_s": secs, "infer_us": us, "params": n_params(P), "status": "ok", "trace": tag})
            except Exception as exc:
                print(f"  {label:<11}{cname:<16}{type(exc).__name__}: {str(exc)[:40]}", flush=True)
                rows.append({"case": label, "config": cname, "n_modes": nm, "channels": C,
                             "train_batch": Btr, "infer_batch": 0, "rel_l2": float("nan"),
                             "train_s": float("nan"), "infer_us": float("nan"), "params": 0,
                             "status": type(exc).__name__, "trace": ""})
            save(rows)
        print(flush=True)
    save(rows)
    subprocess.run(f"cd {OUT} && tar czf /tmp/docs_full_weights.tar.gz *.npz", shell=True)
    subprocess.run(f"cd /tmp && tar czf /tmp/docs_full_prof.tar.gz {os.path.basename(LOGDIR)} {os.path.basename(CSV)}", shell=True)
    print("FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
