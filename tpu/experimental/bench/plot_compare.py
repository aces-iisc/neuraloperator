"""Ground truth against the baseline and our two configurations, with per-pixel error."""
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from ...fft.core import BF, F
from ...fno import data, model
from ..fno.reference import forward as jnp_fwd

W = "tpu/results/baseline"
ROOT = os.path.expanduser("~/data")
CASES = {"Darcy-32": (24, 16, 32), "Darcy-64": (24, 16, 64)}


def rel(p, g):
    ax = tuple(range(1, g.ndim))
    return float(np.sum(np.sqrt(np.sum((p - g) ** 2, axis=ax)))
                 / np.sum(np.sqrt(np.sum(g ** 2, axis=ax))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Darcy-64")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--out", default="../docs-fno/assets")
    args = ap.parse_args()
    C, nm, res = CASES[args.case]
    _, _, xte, yte = data.load_darcy(ROOT, res)
    n = args.rows
    xs = jnp.asarray(np.asarray(xte[:n], np.float32))
    ys = np.asarray(yte[:n])

    load = lambda f: model.load_params(f"{W}/{args.case}_{f}.npz", channels=C,
                                       n_modes=nm, n_layers=4)
    Pj, Po, Pb = load("jnp"), load("ours"), load("ours-bf16")
    preds = [("baseline (jnp.fft)", np.asarray(jnp_fwd(Pj, xs, nm))),
             ("ours float32", np.asarray(model.forward(Po, xs, nm, F))),
             ("ours bfloat16", np.asarray(model.forward(Pb, xs, nm, BF)).astype(np.float32))]
    short = ["baseline", "float32", "bfloat16"]

    ncol = 1 + 2 * len(preds)
    fig, axes = plt.subplots(n, ncol, figsize=(1.95 * ncol, 2.05 * n))
    for r in range(n):
        t = ys[r].squeeze(); lim = np.max(np.abs(t))
        panels = [("ground truth", t, "viridis", lim)]
        panels += [(k, v[r].squeeze(), "viridis", lim) for k, v in preds]
        panels += [(f"|error| {s}", np.abs(v[r].squeeze() - t), "inferno", None)
                   for s, (_, v) in zip(short, preds)]
        for c, (title, img, cm, l) in enumerate(panels):
            ax = axes[r, c]
            kw = {"vmin": 0, "vmax": l} if l else {}
            im = ax.imshow(img, cmap=cm, origin="lower", **kw)
            if r == 0:
                ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            if l is None:
                fig.colorbar(im, ax=ax, fraction=0.046)

    sub = np.asarray(yte[:64]); xb = jnp.asarray(np.asarray(xte[:64], np.float32))
    rj = rel(np.asarray(jnp_fwd(Pj, xb, nm)), sub)
    ro = rel(np.asarray(model.forward(Po, xb, nm, F)), sub)
    rb = rel(np.asarray(model.forward(Pb, xb, nm, BF)).astype(np.float32), sub)
    grid = f"{res}^2" if args.case.startswith("Darcy") else args.case
    fig.suptitle(f"{args.case} — relative $L_2$:  baseline {rj:.4f},  "
                 f"ours float32 {ro:.4f},  ours bfloat16 {rb:.4f}", y=1.0, fontsize=10)
    fig.tight_layout()
    p = os.path.join(args.out, f"pred_compare_{args.case.lower().replace('-', '')}.png")
    fig.savefig(p, dpi=165, bbox_inches="tight")
    print(f"  wrote {p}")
    print(f"  rel L2 (64 samples): baseline {rj:.4f}  f32 {ro:.4f}  bf16 {rb:.4f}")


if __name__ == "__main__":
    main()
