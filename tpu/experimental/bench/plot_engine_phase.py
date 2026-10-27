"""Which real-input transform wins, over the whole (grid, num_modes) plane, on two machines.

The upper panels are the band: one direct partial DFT per engine, which is what an FNO layer
runs. The lower strips are the staged full non-redundant spectrum, which is the regime the
published engine table measured. They are drawn apart because they are different programs, and
the winner is not the same in both.

Marker area carries the margin over the runner-up, because a winner map alone cannot distinguish
a scheme that wins by a factor of two from one that wins by a percent. A hollow marker says the
library transform beat all three at that cell: the shape still names which of ours came first,
but computing the whole spectrum and throwing most of it away was cheaper than computing the
part. That is the second question the plane answers, and it has a different boundary.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

GPU = "tpu/results/gpu5070/engines_phase_gpu.csv"
TPU = "tpu/results/enginephase/engines_phase_tpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/engine_phase.png"
K_COLS = os.environ.get("PHASE_PLOT_K", "256")
ENGINES = ["reim", "coupled", "reduced"]
STYLE = {"reim": ("#2f6fb2", "o"), "coupled": ("#c4622d", "s"), "reduced": ("#3f8f52", "^")}


def load(path, column):
    """{(family, grid, n_modes): {engine: us}} for one width."""
    if not os.path.exists(path):
        return None
    cells = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if row["k_cols"] != K_COLS or not row.get(column):
            continue
        key = (row["family"], int(row["grid"]), int(row["n_modes"]))
        cells[key][row["engine"]] = float(row[column])
    return cells


def area(margin):
    """Marker area from the margin over the runner-up: 1.0x is small, 4x is large."""
    return 28 + 420 * min(np.log2(max(margin, 1.0)), 2.0) / 2.0


def winners(cells, family):
    out = {}
    for (fam, grid, modes), got in cells.items():
        if fam != family:
            continue
        timed = {e: got[e] for e in ENGINES if e in got}
        if len(timed) < 2:
            continue
        order = sorted(timed, key=timed.get)
        out[(grid, modes)] = (order[0], timed[order[1]] / timed[order[0]],
                              min(timed.values()), got.get("native"))
    return out


def panel_band(ax, band, label):
    grids = sorted({g for g, _ in band})
    for (grid, modes), (who, margin, best, native) in band.items():
        colour, marker = STYLE[who]
        beaten = native is not None and native < best
        ax.scatter(np.log2(grid), np.log2(modes), s=area(margin), marker=marker,
                   facecolors="none" if beaten else colour, edgecolors=colour,
                   linewidths=1.4 if beaten else .7, zorder=3)

    # num_modes cannot exceed N, so the upper-left wedge is not a region where some engine
    # lost -- it is a region that does not exist. Drawn rather than left blank.
    lo, hi = np.log2(min(grids)), np.log2(max(grids))
    ax.fill_between([lo - .5, hi + .5], [lo + .5, hi + .5], hi + .6,
                    color="0.85", alpha=.55, hatch="///", edgecolor="0.6", lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=.9, ls=":", zorder=1)

    ax.set_xlim(lo - .5, hi + .5)
    ax.set_ylim(.5, hi + .6)
    steps = [2 ** p for p in range(1, int(hi) + 1)]
    ax.set_yticks([np.log2(m) for m in steps])
    ax.set_yticklabels([str(m) for m in steps], fontsize=8)
    ax.set_ylabel("num_modes")
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=.18, zorder=0)


def panel_staged(ax, staged):
    for (grid, modes), (who, margin, _, _) in staged.items():
        colour, marker = STYLE[who]
        ax.scatter(np.log2(grid), 0, s=area(margin), c=colour, marker=marker,
                   edgecolors="white", linewidths=.7, zorder=3)
        ax.annotate(f"{margin:.2f}x", (np.log2(grid), 0), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=7, color="0.3")
    ax.set_yticks([0])
    ax.set_yticklabels(["N/2"], fontsize=8)
    ax.set_ylim(-.6, .9)
    ax.set_ylabel("staged", fontsize=9)
    ax.grid(alpha=.18, axis="x", zorder=0)


def ticks(ax, grids):
    ax.set_xticks([np.log2(g) for g in grids])
    ax.set_xticklabels([str(g) for g in grids], fontsize=8)


def main():
    # The TPU figures rebuild the DFT matrix inside every timed call, because XLA does not
    # hoist it out of the jit; the GPU ones do not, because the row matrix is cached across
    # calls. PHASE_TPU_COL=mxu_us drops the matrix build and compares the transforms alone,
    # which is what a layer with the matrix hoisted actually runs.
    tpu_col = os.environ.get("PHASE_TPU_COL", "device_us")
    tpu_label = ("TPU v5e, device time from traces" if tpu_col == "device_us"
                 else "TPU v5e, matrix-unit time only (matrix hoisted)")
    panels = [("RTX 5070, device time from a captured graph", load(GPU, "graph_us")),
              (tpu_label, load(TPU, tpu_col))]
    panels = [(label, cells) for label, cells in panels if cells]
    if not panels:
        raise SystemExit("no sweep found; run the phase benches first")

    fig = plt.figure(figsize=(6.4 * len(panels), 6.4))
    spec = fig.add_gridspec(2, len(panels), height_ratios=[5, 1], hspace=.1, wspace=.22)
    for i, (label, cells) in enumerate(panels):
        band, staged = winners(cells, "band"), winners(cells, "staged")
        top, bottom = fig.add_subplot(spec[0, i]), fig.add_subplot(spec[1, i])
        panel_band(top, band, label)
        panel_staged(bottom, staged)
        grids = sorted({g for g, _ in band})
        for ax in (top, bottom):
            ax.set_xlim(np.log2(min(grids)) - .5, np.log2(max(grids)) + .5)
            ticks(ax, grids)
        top.set_xticklabels([])
        bottom.set_xlabel("$N$")

    handles = [Line2D([], [], color=STYLE[e][0], marker=STYLE[e][1], ls="", ms=8,
                      label=e) for e in ENGINES]
    handles += [Line2D([], [], color="0.35", marker="o", ls="", ms=9, mfc="none", mew=1.4,
                       label="hollow: the library transform beat all three"),
                Patch(facecolor="0.85", hatch="///", edgecolor="0.6",
                      label="num_modes > N")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(bottom=.12)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
