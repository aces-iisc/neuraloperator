"""Prediction figures: input, ground truth, baseline and our engine, in the repo example layout."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...fno import data, model
from ...fno.model import forward as ours_forward
from ..fno.reference import forward as jnp_forward

WEIGHTS = "tpu/results/baseline"
ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))

# (label, loader, n_modes, channels) -- mirrors CASES in docs_baseline.py
CASES = {
    "Darcy-32": (lambda: data.load_darcy(ROOT, 32), 16, 24),
    "Darcy-64": (lambda: data.load_darcy(ROOT, 64), 16, 24),
    "NS-128": (lambda: data.load_ns(ROOT, 128, n_train=16, n_test=8), 64, 64),
}


FORWARD = {"ours": ours_forward, "baseline": jnp_forward}


def figure(case, out, n_rows=3):
    loader, n_modes, channels = CASES[case]
    _, _, xte, yte = loader()
    xs, ys = np.asarray(xte[:n_rows]), np.asarray(yte[:n_rows])

    preds = {}
    for engine, fname in (("baseline", f"{case}_jnp.npz"), ("ours", f"{case}_ours.npz")):
        path = os.path.join(WEIGHTS, fname)
        if not os.path.exists(path):
            print(f"  {case}: missing {fname}, skipping"); return None
        P = model.load_params(path, channels=channels, n_modes=n_modes, n_layers=4)
        preds[engine] = np.asarray(FORWARD[engine](P, xs, n_modes))

    n_col = 5
    fig, axes = plt.subplots(n_rows, n_col, figsize=(2.05 * n_col, 2.05 * n_rows))
    titles = ["Input $a(x)$", "Ground truth", "baseline (jnp.fft)", "ours (reim + partial)",
              "|ours $-$ truth|"]
    for r in range(n_rows):
        truth = ys[r].squeeze()
        panels = [xs[r].squeeze(), truth, preds["baseline"][r].squeeze(),
                  preds["ours"][r].squeeze(), np.abs(preds["ours"][r].squeeze() - truth)]
        lim = np.max(np.abs(truth))
        for c, img in enumerate(panels):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            if c == 0:
                ax.imshow(img, cmap="gray", origin="lower")
            elif c == 4:
                m = ax.imshow(img, cmap="inferno", origin="lower")
                fig.colorbar(m, ax=ax, fraction=0.046)
            else:
                ax.imshow(img, cmap="viridis", origin="lower", vmin=-lim, vmax=lim)
            if r == 0:
                ax.set_title(titles[c], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    rel = np.linalg.norm(preds["ours"] - ys) / np.linalg.norm(ys)
    relb = np.linalg.norm(preds["baseline"] - ys) / np.linalg.norm(ys)
    fig.suptitle(f"{case}: relative $L_2$  ours {rel:.4f}   baseline {relb:.4f}", y=1.0, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out, f"pred_{case.lower().replace('-', '')}.png")
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {p}   ours {rel:.4f} | baseline {relb:.4f}")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../docs-fno/assets")
    ap.add_argument("--cases", default="Darcy-32,Darcy-64")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for c in [c.strip() for c in args.cases.split(",")]:
        try:
            figure(c, args.out)
        except Exception as exc:
            print(f"  {c:<10} FAILED: {type(exc).__name__}: {str(exc)[:90]}")


if __name__ == "__main__":
    main()
