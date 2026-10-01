"""Plots the 2D row-method sweep over n_modes x N, each method at its own best batch.

Three panels: absolute cost, cost relative to the banded method, and which method wins
where.  The middle panel is the point -- it puts the n_modes = N diagonal at x = 1, so
the convergence as the band stops being a saving is read straight off the axis.
"""
import csv, math, sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/modesweep/modesweep2d.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tpu/results/modesweep/modesweep2d.png"
METHODS = ["reim", "coupled", "reduced"]
COLOR = {"reim": "#2f6fb2", "coupled": "#2e8b57", "reduced": "#a4406a"}
MARK = {"reim": "o", "coupled": "^", "reduced": "D"}


def load(path):
    """Best batch per (N, n_modes, method), keeping only cells whose sweep finished.

    A method that was cut off mid-batch-sweep would look slow for no reason, so a cell is
    dropped unless every method in it was offered the same batches.
    """
    tried = defaultdict(set)
    seen = {}
    for r in csv.DictReader(open(path)):
        if r["status"] != "ok":
            continue
        us = float(r["us_per_sample"])
        if math.isnan(us):
            continue
        cell = (int(r["grid"]), int(r["n_modes"]), r["method"])
        tried[cell].add(int(r["batch"]))
        if cell not in seen or us < seen[cell][0]:
            seen[cell] = (us, int(r["batch"]))
    best = {}
    for (n, m, meth), val in seen.items():
        peers = [tried[(n, m, o)] for o in METHODS if (n, m, o) in tried]
        if tried[(n, m, meth)] == max(peers, key=len):
            best[(n, m, meth)] = val
    return best


best = load(CSV)
grids = sorted({k[0] for k in best})
fig = plt.figure(figsize=(16.5, 9.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=.32, wspace=.22)

# -- panel A: absolute cost, one line per method, all N on one log-log axis ----
axa = fig.add_subplot(gs[0, 0])
for method in METHODS:
    xs, ys = [], []
    for n in grids:
        for m in sorted({k[1] for k in best if k[0] == n}):
            if (n, m, method) in best:
                xs.append(n * n)
                ys.append(best[(n, m, method)][0])
    order = np.argsort(xs)
    axa.plot(np.array(xs)[order], np.array(ys)[order], MARK[method], ms=4, alpha=.55,
             color=COLOR[method], label=method, lw=0)
axa.set_xscale("log"); axa.set_yscale("log")
axa.set_xlabel("grid points $N^2$"); axa.set_ylabel(r"$\mu$s / sample")
axa.set_title("A. absolute cost, every (N, n_modes) cell")
axa.grid(alpha=.25, which="both"); axa.legend(fontsize=9)

# -- panel B: ratio to reim against n_modes / N -- the diagonal sits at x = 1 --
axb = fig.add_subplot(gs[0, 1])
for method in METHODS:
    if method == "reim":
        continue
    pts = defaultdict(list)
    for (n, m, meth), (us, _) in best.items():
        if meth != method or (n, m, "reim") not in best:
            continue
        pts[m / n].append(us / best[(n, m, "reim")][0])
    xs = sorted(pts)
    ys = [float(np.median(pts[x])) for x in xs]
    axb.plot(xs, ys, MARK[method] + "-", color=COLOR[method], lw=1.8, ms=6, label=method)
axb.axhline(1.0, color="#2f6fb2", lw=2, ls="--", label="reim (reference)")
axb.axvline(1.0, color="#555", lw=1, ls=":")
axb.annotate("full spectrum\n$n_{modes}=N$", xy=(1.0, 1.0), xytext=(.42, 1.30),
             fontsize=9, color="#333",
             arrowprops=dict(arrowstyle="->", color="#555", lw=1))
axb.set_xscale("log"); axb.set_xlabel(r"$n_{modes}\,/\,N$")
axb.set_ylabel(r"time $\div$ reim")
axb.set_title("B. relative cost vs the band fraction (median over N)")
axb.grid(alpha=.25, which="both"); axb.legend(fontsize=9)

# -- panel C: per-N curves of the ratio, so the trend is visible per grid -------
axc = fig.add_subplot(gs[1, 0])
cmap = plt.get_cmap("viridis")
for i, n in enumerate(grids):
    ms = sorted({k[1] for k in best if k[0] == n and (n, k[1], "reim") in best})
    if len(ms) < 2:
        continue
    ratio = []
    for m in ms:
        cands = [best[(n, m, meth)][0] for meth in METHODS if (n, m, meth) in best]
        ratio.append(best[(n, m, "reim")][0] / min(cands))
    axc.plot([m / n for m in ms], ratio, "o-", ms=4, lw=1.6,
             color=cmap(i / max(1, len(grids) - 1)), label=f"N={n}")
axc.axhline(1.0, color="#555", lw=1.5, ls="--")
axc.axvline(1.0, color="#555", lw=1, ls=":")
axc.set_xscale("log"); axc.set_xlabel(r"$n_{modes}\,/\,N$")
axc.set_ylabel("reim $\\div$ best method")
axc.set_title("C. how much reim loses to the best rival, per N\n(>1 means another method won)")
axc.grid(alpha=.25, which="both"); axc.legend(fontsize=8, ncol=2)

# -- panel D: winner map over (N, n_modes) -------------------------------------
axd = fig.add_subplot(gs[1, 1])
all_m = sorted({k[1] for k in best})
grid_img = np.full((len(grids), len(all_m)), np.nan)
for gi, n in enumerate(grids):
    for mi, m in enumerate(all_m):
        cands = [(best[(n, m, meth)][0], meth) for meth in METHODS if (n, m, meth) in best]
        if not cands:
            continue
        grid_img[gi, mi] = METHODS.index(min(cands)[1])
cmap_w = matplotlib.colors.ListedColormap([COLOR[m] for m in METHODS])
axd.imshow(grid_img, cmap=cmap_w, vmin=-.5, vmax=len(METHODS) - .5,
           aspect="auto", origin="lower")
for gi, n in enumerate(grids):
    for mi, m in enumerate(all_m):
        if m == n:
            axd.add_patch(plt.Rectangle((mi - .5, gi - .5), 1, 1, fill=False,
                                        edgecolor="k", lw=2))
        if not np.isnan(grid_img[gi, mi]):
            cands = [(best[(n, m, meth)][0], meth) for meth in METHODS if (n, m, meth) in best]
            cands.sort()
            gap = cands[1][0] / cands[0][0] if len(cands) > 1 else 1.0
            axd.text(mi, gi, f"{gap:.2f}", ha="center", va="center", fontsize=6.5,
                     color="w" if gap < 1.05 else "k")
axd.set_xticks(range(len(all_m))); axd.set_xticklabels(all_m, fontsize=8)
axd.set_yticks(range(len(grids))); axd.set_yticklabels(grids, fontsize=8)
axd.set_xlabel("$n_{modes}$"); axd.set_ylabel("$N$")
axd.set_title("D. winning method per cell (boxed = $n_{modes}=N$)\n"
              "number = runner-up / winner, so 1.00 is a tie")
axd.legend(handles=[plt.Line2D([], [], marker="s", ls="", color=COLOR[m], label=m)
                    for m in METHODS], fontsize=8, loc="lower right", framealpha=.9)

fig.suptitle("2D spectral-conv row methods over the $n_{modes}\\times N$ grid "
             "(f32, precision highest, v5e, each cell at its own best batch)",
             fontsize=12.5, y=.985)
fig.savefig(OUT, dpi=155, bbox_inches="tight")
print("wrote", OUT)
