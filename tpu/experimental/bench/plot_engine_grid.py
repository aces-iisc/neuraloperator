"""Which real-input transform wins over a dense (grid, num_modes) plane, on two machines.

One figure per dimensionality, because they are different experiments: in 1-D the engine is the
whole transform, in 2-D it reaches the row pass only and the column pass is the same work
whichever engine is chosen. Putting them in one frame would invite reading a 2-D margin as if it
were a 1-D one.

  python -m tpu.experimental.bench.plot_engine_grid 1 [--nolib] [--device gpu|tpu|both] [out.png]

With --nolib the library column is dropped and the map answers the narrower question of which
of the three is first, without the separate question of whether any of them was worth using.

Marker area carries the margin over the runner-up, because a winner map alone cannot separate a
scheme that wins by a factor of two from one that wins by a percent. A hollow marker says the
library transform beat all three: the shape still names which of ours came first, but computing
the whole spectrum and discarding most of it was cheaper than computing the part.

The lower strip, on the 1-D figure only, is the staged full non-redundant spectrum -- a
different program from the band, and the regime the published engine table measured.
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

GPU = ("RTX 5070", "tpu/results/gpu5070/engines_grid_gpu.csv", "graph_us")
TPU = ("TPU v5e", "tpu/results/enginephase/engines_grid_tpu.csv", "device_us")
ENGINES = ["reim", "coupled", "reduced"]
STYLE = {"reim": ("#2f6fb2", "o"), "coupled": ("#c4622d", "s"), "reduced": ("#3f8f52", "^")}


def load(path, column, dims):
    """{(family, grid, n_modes): {engine: us}} for one dimensionality."""
    if not os.path.exists(path):
        return None
    cells = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if int(row["dims"]) != dims or not row.get(column):
            continue
        key = (row["family"], int(row["grid"]), int(row["n_modes"]))
        cells[key][row["engine"]] = float(row[column])
    return cells or None


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


def area(margin):
    """Marker area from the margin over the runner-up: 1.0x small, 4x large."""
    return 18 + 300 * min(np.log2(max(margin, 1.0)), 2.0) / 2.0


def panel_band(ax, band, label, show_lib=True):
    grids = sorted({g for g, _ in band})
    lo, hi = np.log2(min(grids)), np.log2(max(grids))
    # num_modes cannot exceed N, so the upper-left wedge is not a region where some engine
    # lost -- it is a region that does not exist. Drawn rather than left blank.
    ax.fill_between([lo - .5, hi + .5], [lo, hi + 1.0], hi + 1.0,
                    color="0.85", alpha=.55, hatch="///", edgecolor="0.6", lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=.9, ls=":", zorder=1)
    for (grid, modes), (who, margin, best, native) in band.items():
        colour, marker = STYLE[who]
        beaten = show_lib and native is not None and native < best
        ax.scatter(np.log2(grid), np.log2(modes), s=area(margin), marker=marker,
                   facecolors="none" if beaten else colour, edgecolors=colour,
                   linewidths=1.1 if beaten else .5, zorder=3)
    ax.set_xlim(lo - .5, hi + .5)
    ax.set_ylim(.4, hi + .7)
    steps = [2 ** p for p in range(1, int(hi) + 1)]
    ax.set_yticks([np.log2(m) for m in steps])
    ax.set_yticklabels([str(m) for m in steps], fontsize=8)
    ax.set_ylabel("num_modes")
    ax.set_title(label, fontsize=11)
    ax.grid(alpha=.18, zorder=0)


def panel_staged(ax, staged):
    for (grid, _), (who, margin, _, _) in staged.items():
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


def main():
    args = sys.argv[1:]
    show_lib = "--nolib" not in args
    which = "both"
    if "--device" in args:
        which = args[args.index("--device") + 1]
    argv = [a for a in args if a != "--nolib"]
    if "--device" in argv:
        at = argv.index("--device")
        del argv[at:at + 2]
    dims = int(argv[0]) if argv else 1
    sources = {"gpu": [GPU], "tpu": [TPU], "both": [GPU, TPU]}[which]
    tag = "" if which == "both" else f"_{which}"
    suffix = "" if show_lib else "_nolib"
    out = argv[1] if len(argv) > 1 else f"tpu/results/engine_grid{tag}_{dims}d{suffix}.png"
    panels = [(f"{name}, {dims}-D", load(path, column, dims))
              for name, path, column in sources]
    panels = [(label, cells) for label, cells in panels if cells]
    if not panels:
        raise SystemExit(f"no {dims}-D sweep found; run the grid benches first")

    show_staged = any(winners(cells, "staged") for _, cells in panels)
    rows = 2 if show_staged else 1
    width = 6.6 * len(panels) if len(panels) > 1 else 7.4
    fig = plt.figure(figsize=(width, 6.6 if show_staged else 5.8))
    spec = fig.add_gridspec(rows, len(panels),
                            height_ratios=[5, 1] if show_staged else [1],
                            hspace=.1, wspace=.22)

    for i, (label, cells) in enumerate(panels):
        band = winners(cells, "band")
        top = fig.add_subplot(spec[0, i])
        panel_band(top, band, label, show_lib)
        grids = sorted({g for g, _ in band})
        axes = [top]
        if show_staged:
            bottom = fig.add_subplot(spec[1, i])
            panel_staged(bottom, winners(cells, "staged"))
            axes.append(bottom)
            top.set_xticklabels([])
        for ax in axes:
            ax.set_xlim(np.log2(min(grids)) - .5, np.log2(max(grids)) + .5)
            ax.set_xticks([np.log2(g) for g in grids])
            ax.set_xticklabels([str(g) for g in grids], fontsize=8)
        axes[-1].set_xlabel("$N$")

    handles = [Line2D([], [], color=STYLE[e][0], marker=STYLE[e][1], ls="", ms=8, label=e)
               for e in ENGINES]
    if show_lib:
        handles += [Line2D([], [], color="0.35", marker="o", ls="", ms=8, mfc="0.35",
                           label="filled: it also beat the library"),
                    Line2D([], [], color="0.35", marker="o", ls="", ms=8, mfc="none", mew=1.3,
                           label="hollow, any shape: the library beat all three")]
    handles.append(Patch(facecolor="0.85", hatch="///", edgecolor="0.6", label="num_modes > N"))
    columns = len(handles) if len(panels) > 1 else 3
    fig.legend(handles=handles, loc="lower center", ncol=columns, fontsize=9, frameon=False,
               bbox_to_anchor=(.5, -.005))
    fig.subplots_adjust(bottom=.12)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
