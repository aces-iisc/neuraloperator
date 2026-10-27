"""Loaders. Each returns (x_train, y_train, x_test, y_test) as float32 (N, 1, H, W) arrays,
standardised by the TRAIN mean/std."""
import os

import numpy as np
import torch


def _std(xtr, ytr, xte, yte):
    xm, xs = xtr.mean(), xtr.std() + 1e-8
    ym, ys = ytr.mean(), ytr.std() + 1e-8
    return (xtr - xm) / xs, (ytr - ym) / ys, (xte - xm) / xs, (yte - ym) / ys


def load_darcy(root, res=32):
    tr = torch.load(os.path.join(root, f"darcy_train_{res}.pt"), weights_only=False)
    te = torch.load(os.path.join(root, f"darcy_test_{res}.pt"), weights_only=False)
    g = lambda d: (np.asarray(d["x"], np.float32)[:, None],
                   np.asarray(d["y"], np.float32)[:, None])
    return _std(*g(tr), *g(te))


def load_ns(root, res=128, n_train=10000, n_test=1000):
    tr = torch.load(os.path.join(root, f"nsforcing_train_{res}.pt"), weights_only=False)
    te = torch.load(os.path.join(root, f"nsforcing_test_{res}.pt"), weights_only=False)

    def g(d, n):
        x = np.asarray(d["x"], np.float32)[:n]
        y = np.asarray(d["y"], np.float32)[:n]
        x = x[:, None] if x.ndim == 3 else x[:, :1]
        y = y[:, None] if y.ndim == 3 else y[:, :1]
        return x, y
    return _std(*g(tr, n_train), *g(te, n_test))


def load_burgers(root):
    tr = torch.load(os.path.join(root, "burgers_train_16.pt"), weights_only=False)
    te = torch.load(os.path.join(root, "burgers_test_16.pt"), weights_only=False)

    def build(d):
        x0 = np.asarray(d["x"], np.float32)
        y = np.asarray(d["y"], np.float32)
        x = np.broadcast_to(x0[:, None, :], (x0.shape[0], y.shape[1], x0.shape[1]))
        return x[:, None], y[:, None]
    return _std(*build(tr), *build(te))
