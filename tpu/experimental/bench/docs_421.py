"""Darcy-421: train once with our f32 engine, then time every engine on those same weights."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import BF, F
from ...fno import data, model, train
from ...fno.layers import n_params
from ..fno.reference import forward as jnp_fwd

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
CSV = os.environ.get("DOCS_421_CSV", "/tmp/docs_421.csv")
LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_421")
OUT = os.environ.get("DOCS_421_OUT", "/tmp/docs_421")
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "20"))
NTRAIN = int(os.environ.get("DOCS_NTRAIN", "1000"))
NTEST = int(os.environ.get("DOCS_NTEST", "200"))
NM, CH, BTR = 16, 24, 8
FIELDS = ["phase", "engine", "batch", "value", "unit", "rel_l2", "status", "trace"]


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


def step_time(fwd, P, x, y, dt, reps=12):
    """Seconds per training step: forward, loss, backward. Not a full training run."""
    def loss(Q):
        pred = fwd(Q, x, NM) if fwd is jnp_fwd else fwd(Q, x, NM, dt)
        return jnp.mean((pred - y) ** 2)
    g = jax.jit(jax.value_and_grad(loss))
    jax.block_until_ready(g(P))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(g(P)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[reps//2]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"Darcy-421  n_train={NTRAIN} n_test={NTEST} epochs={EPOCHS} "
          f"num_modes={NM} channels={CH} batch={BTR}", flush=True)
    os.makedirs(OUT, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    xtr, ytr, xte, yte = data.load_darcy(ROOT, 421)
    xtr, ytr = xtr[:NTRAIN], ytr[:NTRAIN]
    xte, yte = xte[:NTEST], yte[:NTEST]
    print(f"train{tuple(xtr.shape)} test{tuple(xte.shape)}", flush=True)
    rows = []

    # --- per-step training cost of each engine, on identical weights ---
    P0 = model.__dict__["layers"].init_fno(jax.random.PRNGKey(0), 1, 1, channels=CH,
                                           n_modes=NM, n_layers=4) \
         if hasattr(model, "layers") else None
    from ...fno import layers as L
    P0 = L.init_fno(jax.random.PRNGKey(0), 1, 1, channels=CH, n_modes=NM, n_layers=4)
    xb = jnp.asarray(np.asarray(xtr[:BTR], np.float32))
    yb = jnp.asarray(np.asarray(ytr[:BTR], np.float32))
    print(f"\n  {'phase':<14}{'engine':<16}{'value':>12}{'unit':>10}", flush=True)
    base_step = None
    for name, fwd, dt in (("baseline", jnp_fwd, F), ("ours f32", model.forward, F),
                          ("ours bf16", model.forward, BF)):
        try:
            s = step_time(fwd, P0, xb, yb, dt)
            if base_step is None: base_step = s
            print(f"  {'train step':<14}{name:<16}{s*1e3:>12.2f}{'ms':>10}"
                  f"   {base_step/s:.2f}x", flush=True)
            rows.append({"phase": "train_step", "engine": name, "batch": BTR, "value": s*1e3,
                         "unit": "ms", "rel_l2": float("nan"), "status": "ok", "trace": ""})
        except Exception as exc:
            print(f"  {'train step':<14}{name:<16}{type(exc).__name__}", flush=True)
            rows.append({"phase": "train_step", "engine": name, "batch": BTR,
                         "value": float("nan"), "unit": "ms", "rel_l2": float("nan"),
                         "status": type(exc).__name__, "trace": ""})
        save(rows)

    # --- train ONCE, with our f32 engine ---
    print(f"\n  training once with ours f32 ...", flush=True)
    t0 = time.perf_counter()
    P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=NM, channels=CH, n_layers=4,
                               steps=EPOCHS, batch=BTR, verbose=False,
                               save=os.path.join(OUT, "Darcy-421_ours_f32.npz"))
    print(f"  trained in {secs:.1f} s, rel L2 = {rel:.4e}, params = {n_params(P):,}", flush=True)
    rows.append({"phase": "train", "engine": "ours f32", "batch": BTR, "value": secs,
                 "unit": "s", "rel_l2": rel, "status": "ok", "trace": ""})
    save(rows)

    # --- inference: every engine on THOSE weights ---
    print(f"\n  {'phase':<14}{'engine':<16}{'batch':>6}{'us/sample':>12}{'rel L2':>12}", flush=True)
    for name, fwd, dt, B in (("baseline", jnp_fwd, F, 8), ("ours f32", model.forward, F, 8),
                             ("ours bf16", model.forward, BF, 8),
                             ("ours f32", model.forward, F, 4),
                             ("ours bf16", model.forward, BF, 4)):
        n = (len(xte)//B)*B
        tag = f"421_{name.replace(' ','_')}_{B}"
        try:
            f = jax.jit(lambda z, _f=fwd, _d=dt: _f(P, z, NM) if _f is jnp_fwd else _f(P, z, NM, _d))
            x = jnp.asarray(np.asarray(xte[:B], np.float32))
            jax.block_until_ready(f(x))
            ts = []
            with jax.profiler.trace(os.path.join(LOGDIR, tag)):
                for _ in range(20):
                    t0 = time.perf_counter(); jax.block_until_ready(f(x))
                    ts.append(time.perf_counter()-t0)
            us = sorted(ts)[10]*1e6/B
            rl = rel_l2(f, xte, yte, B, n)
            print(f"  {'inference':<14}{name:<16}{B:>6}{us:>12.2f}{rl:>12.4e}", flush=True)
            rows.append({"phase": "infer", "engine": name, "batch": B, "value": us,
                         "unit": "us/sample", "rel_l2": rl, "status": "ok", "trace": tag})
        except Exception as exc:
            print(f"  {'inference':<14}{name:<16}{B:>6}  {type(exc).__name__}", flush=True)
            rows.append({"phase": "infer", "engine": name, "batch": B, "value": float("nan"),
                         "unit": "us/sample", "rel_l2": float("nan"),
                         "status": type(exc).__name__, "trace": tag})
        save(rows)
    save(rows)
    os.system("cd /tmp && tar czf /tmp/p421.tar.gz docs_prof_421 docs_421.csv docs_421")
    print("\nD421_DONE", flush=True)


if __name__ == "__main__":
    main()
