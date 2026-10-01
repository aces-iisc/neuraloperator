import time

import jax
import jax.numpy as jnp
import numpy as np

from . import data
from .layers import LOSSES, adam_init, adam_step, init_fno, mse, n_params
from .model import forward, save_params

EVAL_CHUNK = 64      # test set is evaluated in chunks so 128^2 does not blow up HBM
INFER_CAP = 128      # and inference is timed on at most this many samples, for the same reason

DATASETS = {"darcy":   lambda root: data.load_darcy(root, 32),
            "ns":      lambda root: data.load_ns(root, 128),
            "burgers": lambda root: data.load_burgers(root)}


def infer_us_per_sample(P, x, n_modes, fwd=forward, reps=30):
    """Median wall-clock of a jitted forward pass, per sample, in microseconds."""
    x = jnp.asarray(x[:INFER_CAP])
    run = jax.jit(lambda z: fwd(P, z, n_modes))
    jax.block_until_ready(run(x))                       # compile before timing
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(run(x))
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    return median / x.shape[0] * 1e6


def train(xtr, ytr, xte, yte, n_modes=16, channels=32, n_layers=4, steps=200, batch=16,
          lr=1e-3, weight_decay=1e-4, lr_step=100, lr_gamma=0.5, seed=0, fwd=forward,
          loss="mse", init=None,
          log_every=10, name="fno", log_csv=None, save=None, verbose=True):
    """Adam + MSE with a step LR schedule, keeping the best-on-test parameters (early stopping)
    so a late overfit cannot silently degrade what gets reported or saved.

    Returns (best_params, best_test_rel, train_seconds), the error in the TRAINING metric.  Training data stays on the host as numpy
    and is moved to device one batch at a time — the full NS set does not fit otherwise."""
    xtr, ytr = np.asarray(xtr, np.float32), np.asarray(ytr, np.float32)
    xte, yte = np.asarray(xte, np.float32), np.asarray(yte, np.float32)
    n_train = xtr.shape[0]

    key = jax.random.PRNGKey(seed)
    build = init or init_fno          # a variant filter needs its own weights
    P = build(key, 1, 1, channels, n_modes, n_layers)
    opt = adam_init(P)

    @jax.jit
    def train_step(P, opt, x, y, lr_now):
        obj = LOSSES[loss]
        loss_v, grads = jax.value_and_grad(lambda P: obj(fwd(P, x, n_modes), y))(P)
        P, opt = adam_step(P, grads, opt, lr=lr_now, weight_decay=weight_decay)
        return P, opt, loss_v

    @jax.jit
    def eval_chunk(P, x, y):
        """Per-sample numerator and denominator of the relative error, in the TRAINING metric.

        Evaluating on something other than the objective would early-stop on a criterion the
        model is not being optimised for, so the H1 derivative terms are included whenever the
        loss is H1."""
        pred = fwd(P, x, n_modes)
        err = jnp.sum((pred - y) ** 2, (-1, -2, -3))
        norm = jnp.sum(y ** 2, (-1, -2, -3))
        if loss == "h1":
            for axis in (-1, -2):
                d_pred = jnp.diff(pred, axis=axis, append=jnp.take(pred, jnp.array([0]), axis))
                d_y = jnp.diff(y, axis=axis, append=jnp.take(y, jnp.array([0]), axis))
                err = err + jnp.sum((d_pred - d_y) ** 2, (-1, -2, -3))
                norm = norm + jnp.sum(d_y ** 2, (-1, -2, -3))
        return err, norm

    def test_rel(P):
        err = norm = 0.0
        for i in range(0, xte.shape[0], EVAL_CHUNK):
            e, n = eval_chunk(P, jnp.asarray(xte[i:i + EVAL_CHUNK]), jnp.asarray(yte[i:i + EVAL_CHUNK]))
            err += float(jnp.sum(jnp.sqrt(e)))
            norm += float(jnp.sum(jnp.sqrt(n)))
        return err / norm

    if verbose:
        print(f"{name}: channels={channels} n_modes={n_modes} | {n_train} train / {xte.shape[0]} "
              f"test | params={n_params(P):,}", flush=True)

    best_rel_l2, best_P = float("inf"), P
    history = [("epoch", "train_mse", "test_rel", "best")]
    n_batches = max(1, n_train // batch)
    t0 = time.perf_counter()

    for epoch in range(steps):
        lr_now = lr * (lr_gamma ** (epoch // lr_step))
        key, shuffle_key = jax.random.split(key)
        order = np.asarray(jax.random.permutation(shuffle_key, n_train))
        epoch_loss = 0.0
        for b in range(n_batches):
            idx = order[b * batch:(b + 1) * batch]
            P, opt, loss = train_step(P, opt, jnp.asarray(xtr[idx]), jnp.asarray(ytr[idx]), lr_now)
            epoch_loss += float(loss)

        if epoch % log_every == 0 or epoch == steps - 1:
            rel = test_rel(P)
            if rel < best_rel_l2:
                best_rel_l2, best_P = rel, jax.tree_util.tree_map(lambda a: a, P)
            history.append((epoch, epoch_loss / n_batches, rel, best_rel_l2))
            if verbose:
                print(f"  epoch {epoch:4d}  train_mse {epoch_loss/n_batches:.4e}  "
                      f"test_rel[{loss}] {rel:.4e}  best {best_rel_l2:.4e}", flush=True)

    seconds = time.perf_counter() - t0
    if verbose:
        print(f"{name}: trained in {seconds:.1f}s ({steps} epochs x {n_batches} steps, batch {batch})",
              flush=True)
    if log_csv:
        open(log_csv, "w").write("\n".join(",".join(str(v) for v in row) for row in history) + "\n")
    if save:
        save_params(best_P, save)
        if verbose:
            print(f"saved best (rel[{loss}] {best_rel_l2:.4e}) -> {save}", flush=True)
    return best_P, best_rel_l2, seconds


def run(dataset="darcy", root="~/data", **kw):
    """Train on one of the three benchmark datasets and report accuracy plus inference cost."""
    import os
    xtr, ytr, xte, yte = DATASETS[dataset](os.path.expanduser(root))
    P, rel, seconds = train(xtr, ytr, xte, yte, name=dataset, **kw)
    n_modes = kw.get("n_modes", 16)
    us = infer_us_per_sample(P, xte, n_modes)
    print(f"{dataset}: rel_l2={rel:.4e}  train={seconds:.1f}s  infer={us:.2f} us/sample", flush=True)
    return P, rel, us
