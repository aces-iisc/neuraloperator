"""Training loop: AdamW, step LR, early stopping on the training metric."""
import time

import numpy as np
import torch

from ..fft.core import F
from .layers import LOSSES
from .model import FNO

EVAL_CHUNK = 64


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def infer_us_per_sample(module, x, reps=30, device=None, cap=128):
    """Median wall clock of a forward pass, per sample, in microseconds."""
    device = device or next(module.parameters()).device
    x = torch.as_tensor(np.asarray(x[:cap], np.float32)).to(device)
    module.eval()
    module(x)
    _sync(device)
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        module(x)
        _sync(device)
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2] / x.shape[0] * 1e6


@torch.no_grad()
def evaluate(module, xte, yte, loss="h1", device=None, chunk=EVAL_CHUNK):
    """Mean error per sample. The loss sums over the batch, so divide by the split size."""
    device = device or next(module.parameters()).device
    obj, total = LOSSES[loss], 0.0
    module.eval()
    for i in range(0, len(xte), chunk):
        x = torch.as_tensor(np.asarray(xte[i:i + chunk], np.float32)).to(device)
        y = torch.as_tensor(np.asarray(yte[i:i + chunk], np.float32)).to(device)
        total += float(obj(module(x), y))
    return total / len(xte)


def train(xtr, ytr, xte, yte, n_modes=16, channels=32, n_layers=4, filter="dense",
          epochs=200, batch=16, lr=1e-3, weight_decay=1e-4, lr_step=100, lr_gamma=0.5,
          seed=0, loss="h1", device=None, compile=False, name="fno", save=None, verbose=True):
    """Returns (module, best_error, seconds). Data stays on the host and moves a batch at a
    time; the full NS set does not fit otherwise."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)

    xtr = np.asarray(xtr, np.float32)
    ytr = np.asarray(ytr, np.float32)
    module = FNO(1, 1, channels, n_modes, n_layers, filter).to(device)
    run = torch.compile(module) if compile else module

    # fused AdamW keeps the parameter update in one kernel instead of one per tensor
    fused = device.type == "cuda"
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=weight_decay, fused=fused)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    objective = LOSSES[loss]

    n_train = xtr.shape[0]
    n_batches = max(1, n_train // batch)
    best_error, best_state = float("inf"), None
    pin = device.type == "cuda"
    start = time.perf_counter()

    if verbose:
        print(f"{name}: channels={channels} n_modes={n_modes} filter={filter} | {n_train} train"
              f" / {len(xte)} test | params={module.n_params():,} | {device}", flush=True)

    for epoch in range(epochs):
        module.train()
        order = np.random.default_rng(seed + epoch).permutation(n_train)
        running = 0.0
        for b in range(n_batches):
            idx = order[b * batch:(b + 1) * batch]
            x = torch.from_numpy(xtr[idx]).pin_memory() if pin else torch.from_numpy(xtr[idx])
            y = torch.from_numpy(ytr[idx]).pin_memory() if pin else torch.from_numpy(ytr[idx])
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            opt.zero_grad(set_to_none=True)
            value = objective(run(x), y) / x.shape[0]
            value.backward()
            opt.step()
            running += float(value)
        sched.step()

        error = evaluate(module, xte, yte, loss, device)
        if error < best_error:
            best_error = error
            best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:4d}  train {running / n_batches:.4e}  "
                  f"test[{loss}] {error:.4e}  best {best_error:.4e}", flush=True)

    seconds = time.perf_counter() - start
    if best_state is not None:
        module.load_state_dict(best_state)
    if save:
        torch.save(module.state_dict(), save)
    if verbose:
        print(f"{name}: trained in {seconds:.1f}s, best test[{loss}] {best_error:.4e}",
              flush=True)
    return module, best_error, seconds
