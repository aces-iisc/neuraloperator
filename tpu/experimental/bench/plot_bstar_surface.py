"""How much batch each spectral shape needs, over the (grid, num_modes) plane.

  python -m tpu.experimental.bench.plot_bstar_surface map  [gpu|tpu] [1|2] [out.png]
  python -m tpu.experimental.bench.plot_bstar_surface collapse [out.png]

The map prints B* per cell, one panel per engine, with the colour on a log batch scale so a
factor of two reads as one step.  A cell whose knee sits past the widest batch measured is
drawn hollow and labelled with a leading sign, because that number is a lower bound on B* and
not a measurement of it.

The collapse plots the same B* against the per-sample element count.  If a single curve fits,
then B* is a function of the working set alone and the autotuner needs no per-shape search;
where the curve splits by device or by dimensionality, it does.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tpu.experimental.bench.bstar_surface import DEVICES, ENGINES, surface, working_set

STYLE = {"reim": ("#2f6fb2", "o"), "coupled": ("#c4622d", "s"),
         "reduced": ("#3f8f52", "^"), "native": ("#7a5ea8", "D")}
FAMILY = "band"


# The two devices were swept on different mode ladders: the GPU record steps in powers of two
# and the TPU record predates that and steps in thirty-three irregular values.  The map keeps
# the powers of two, which both ladders contain, so a column means the same shape on either
# device and an empty cell means the shape does not exist rather than was not swept.
def power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def axes_of(cells):
    grids = sorted({grid for _, grid, _, _ in cells})
    modes = sorted({n_modes for _, _, n_modes, _ in cells})
    return grids, modes


def draw_map(device, dims, out_path):
    cells = {key: knee for key, knee in surface(device, dims).items()
             if key[0] == FAMILY and power_of_two(key[2])}
    if not cells:
        raise SystemExit(f"no {FAMILY} cells for {device} {dims}-D")
    grids, modes = axes_of(cells)
    engines = [e for e in ENGINES if any(key[3] == e for key in cells)]

    figure, panels = plt.subplots(1, len(engines), figsize=(4.1 * len(engines), 4.6),
                                  sharex=True, sharey=True)
    panels = np.atleast_1d(panels)
    values = [np.log2(knee.b_star) for knee in cells.values()]
    low, high = min(values), max(values)

    for panel, engine in zip(panels, engines):
        grid_of = np.full((len(grids), len(modes)), np.nan)
        for (_, grid, n_modes, name), knee in cells.items():
            if name == engine:
                grid_of[grids.index(grid), modes.index(n_modes)] = np.log2(knee.b_star)
        panel.imshow(grid_of, origin="lower", aspect="auto", cmap="viridis",
                     vmin=low, vmax=high, interpolation="nearest")
        for (_, grid, n_modes, name), knee in cells.items():
            if name != engine:
                continue
            row, column = grids.index(grid), modes.index(n_modes)
            label = _short(knee.b_star) + ("+" if knee.censored else "")
            shade = "white" if np.log2(knee.b_star) < (low + high) / 2 else "black"
            panel.text(column, row, label, ha="center", va="center", fontsize=6,
                       color=shade, style="italic" if knee.censored else "normal")
            if knee.censored:
                panel.add_patch(plt.Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False,
                                              edgecolor="white", linewidth=0.9, hatch="///"))
        panel.set_title(engine, fontsize=10, color=STYLE[engine][0])
        panel.set_xticks(range(len(modes)))
        panel.set_xticklabels(modes, fontsize=6, rotation=90)
        panel.set_xlabel("num_modes $m$")

    panels[0].set_yticks(range(len(grids)))
    panels[0].set_yticklabels(grids, fontsize=7)
    panels[0].set_ylabel("grid $N$")
    bar = figure.colorbar(panels[0].images[0], ax=panels.tolist(), pad=0.02)
    bar.set_label("$\\log_2 B^{*}$")
    figure.suptitle(f"{DEVICES[device][0]}, {dims}-D, {FAMILY}", fontsize=11)
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


# Batches reach five digits and the cells are narrow, so anything past a thousand is written
# in k.  A trailing "+" marks a censored cell: B* is at least this, and the sweep stopped there.
def _short(batch):
    if batch < 1000:
        return str(batch)
    thousands = batch / 1000.0
    return f"{thousands:.0f}k" if thousands >= 10 or thousands == int(thousands) else f"{thousands:.1f}k"


def draw_collapse(out_path):
    figure, panels = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)
    for panel, dims in zip(panels, (1, 2)):
        for device, marker in (("gpu", "o"), ("tpu", "^")):
            cells = {k: v for k, v in surface(device, dims).items()
                     if k[0] == FAMILY and k[3] == "reim"}
            resolved = [(working_set(dims, g, m), knee.b_star)
                        for (_, g, m, _), knee in cells.items() if not knee.censored]
            bounded = [(working_set(dims, g, m), knee.b_star)
                       for (_, g, m, _), knee in cells.items() if knee.censored]
            colour = "#2f6fb2" if device == "gpu" else "#c4622d"
            if resolved:
                panel.scatter(*zip(*resolved), s=26, marker=marker, color=colour,
                              label=f"{DEVICES[device][0]}", zorder=3)
            if bounded:
                panel.scatter(*zip(*bounded), s=26, marker=marker, facecolors="none",
                              edgecolors=colour, linewidths=0.8, zorder=2)
        panel.set_xscale("log", base=2)
        panel.set_yscale("log", base=2)
        panel.set_xlabel("per-sample elements  $Nm$" if dims == 1 else "per-sample elements  $N^{2}m$")
        panel.set_title(f"{dims}-D", fontsize=10)
        panel.grid(alpha=0.25, which="both", linewidth=0.4)
        panel.legend(fontsize=8, frameon=False)
    panels[0].set_ylabel("$B^{*}$")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


def main(argv):
    mode = argv[0] if argv else "map"
    out_dir = "tpu/results/bstar_surface"
    os.makedirs(out_dir, exist_ok=True)
    if mode == "collapse":
        draw_collapse(argv[1] if len(argv) > 1 else os.path.join(out_dir, "bstar_collapse.png"))
        return
    if len(argv) >= 3:
        device, dims = argv[1], int(argv[2])
        draw_map(device, dims, argv[3] if len(argv) > 3
                 else os.path.join(out_dir, f"bstar_map_{device}_{dims}d.png"))
        return
    for device in DEVICES:
        for dims in (1, 2):
            draw_map(device, dims, os.path.join(out_dir, f"bstar_map_{device}_{dims}d.png"))


if __name__ == "__main__":
    main(sys.argv[1:])
