"""Field figures read off the dataset files: Darcy, Burgers, Navier-Stokes and the spectrum."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DATA = os.path.expanduser("~/data")
REPO_DATA = "neuralop/data/datasets/data"
DPI = 170


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _clean(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def darcy(out, res=128, sample=0):
    """Permeability a(x) beside the pressure u(x) it produces -- the pair the operator maps."""
    d = load(f"{DATA}/darcy_train_{res}.pt") if os.path.exists(f"{DATA}/darcy_train_{res}.pt") \
        else load(f"{REPO_DATA}/darcy_train_{res}.pt")
    a, u = np.asarray(d["x"][sample]), np.asarray(d["y"][sample])
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0))
    im0 = axes[0].imshow(a, cmap="cividis", origin="lower")
    _clean(axes[0], f"permeability $a(x)$  ({res}$\\times${res})")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(u, cmap="magma", origin="lower")
    _clean(axes[1], "pressure $u(x)$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    p = f"{out}/field_darcy_{res}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p, a.shape


def burgers(out, sample=0, n_curves=6):
    """The 1+1D trajectory two ways: stacked waves in time, and the same field as a heatmap."""
    d = load(f"{REPO_DATA}/burgers_train_16.pt")
    traj = np.asarray(d["y"][sample])              # (time, space)
    n_t, n_x = traj.shape
    x = np.linspace(0, 1, n_x)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))
    cmap = plt.get_cmap("viridis")
    for j, t in enumerate(np.linspace(0, n_t - 1, n_curves, dtype=int)):
        axes[0].plot(x, traj[t], color=cmap(j / max(n_curves - 1, 1)), lw=1.6,
                     label=f"$t_{{{t}}}$")
    axes[0].set_xlabel("$x$", fontsize=9); axes[0].set_ylabel("$u(t,x)$", fontsize=9)
    axes[0].set_title(f"Burgers trajectory, {n_curves} of {n_t} steps", fontsize=9)
    axes[0].legend(fontsize=6, ncol=2, frameon=False)
    axes[0].tick_params(labelsize=7)
    im = axes[1].imshow(traj, aspect="auto", cmap="RdBu_r", origin="lower",
                        extent=[0, 1, 0, n_t - 1])
    axes[1].set_xlabel("$x$", fontsize=9); axes[1].set_ylabel("$t$", fontsize=9)
    axes[1].set_title(f"the same field, {n_t}$\\times${n_x} grid", fontsize=9)
    axes[1].tick_params(labelsize=7)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    p = f"{out}/field_burgers.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p, traj.shape


def navier_stokes(out, samples=(0, 1, 2)):
    """Vorticity in and out. The dataset stores w(t) -> w(t+1) pairs, not a full time series."""
    d = load(f"{DATA}/nsforcing_train_128.pt")
    fig, axes = plt.subplots(2, len(samples), figsize=(2.2 * len(samples), 4.4))
    for c, s in enumerate(samples):
        for r, (key, lab) in enumerate((("x", "$w(t)$"), ("y", "$w(t{+}1)$"))):
            w = np.asarray(d[key][s])
            lim = np.max(np.abs(w))
            axes[r, c].imshow(w, cmap="RdBu_r", origin="lower", vmin=-lim, vmax=lim)
            _clean(axes[r, c], f"{lab}  sample {s}" if r == 0 else lab)
    fig.tight_layout()
    p = f"{out}/field_ns128.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p, tuple(np.asarray(d["x"][0]).shape)


def spectrum(out, res=128):
    """Why truncation works: radially averaged spectral energy, with the kept band marked."""
    path = f"{DATA}/darcy_train_{res}.pt"
    d = load(path) if os.path.exists(path) else load(f"{REPO_DATA}/darcy_train_{res}.pt")
    u = np.asarray(d["y"][:64], dtype=np.float64)
    spec = np.abs(np.fft.fft2(u, axes=(1, 2)))**2
    n = spec.shape[1]
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij")
    kr = np.sqrt(kx**2 + ky**2).astype(int)
    prof = np.array([spec[:, kr == k].mean() for k in range(1, n // 2)])
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.loglog(np.arange(1, n // 2), prof / prof[0], lw=1.7, color="#1f4e79")
    ax.axvline(8, color="crimson", ls="--", lw=1.2)
    ax.text(8.6, 0.25, "kept band\n(\\texttt{num\\_modes}/2)", fontsize=7, color="crimson")
    ax.set_xlabel("radial wavenumber $|k|$", fontsize=9)
    ax.set_ylabel("normalised energy", fontsize=9)
    ax.set_title(f"Darcy-{res} spectral energy, mean of 64 samples", fontsize=9)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = f"{out}/field_spectrum_{res}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    return p, prof.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../docs-fno/assets")
    ap.add_argument("--darcy-res", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for fn, kw in ((darcy, {"res": args.darcy_res}), (burgers, {}),
                   (navier_stokes, {}), (spectrum, {"res": args.darcy_res})):
        try:
            p, shape = fn(args.out, **kw)
            print(f"  wrote {p}   from data shaped {shape}")
        except Exception as exc:
            print(f"  {fn.__name__:<14} SKIPPED: {type(exc).__name__}: {str(exc)[:70]}")


if __name__ == "__main__":
    main()
