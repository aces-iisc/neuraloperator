"""The three real-input transforms against num_modes, from a narrow band to the full spectrum.

Two rows, one per dimensionality, because the schemes can only reach one axis of a separable
2-D transform: both exploit the input being real, and by the second pass the field is complex.
The 2-D row therefore shows the whole two-pass transform, of which the engine choice touches
only the first pass.

Priced against the plain real/imag transform, so one is that transform and a point below one is
a packing scheme that beat it. The right-hand end of every panel is num_modes = N, the full
spectrum, which is the regime the schemes were designed for and the one an operator never runs.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "tpu/results/gpu5070/engines_full_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/gpu_engines_full.png"
STYLE = {"coupled": "-", "reduced": "--"}
COLOURS = {32: "#2f6fb2", 64: "#c4622d", 128: "#3f8f52"}


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no sweep at {path}; run engines_full_gpu on the GPU first")
    at = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if row["status"] != "ok":
            continue
        at[(int(row["dims"]), int(row["grid"]), row["engine"])][int(row["n_modes"])] = \
            float(row["us"])
    return at


def main():
    at = load(SWEEP)
    grids = sorted({g for _, g, _ in at})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    for ax, dims, title in ((axes[0], 1, "one dimension, the row transform"),
                            (axes[1], 2, "two dimensions, the whole transform")):
        for grid in grids:
            base = at.get((dims, grid, "reim"), {})
            for engine, style in STYLE.items():
                curve = at.get((dims, grid, engine), {})
                xs = sorted(set(curve) & set(base))
                if not xs:
                    continue
                ax.plot([x / grid for x in xs], [curve[x] / base[x] for x in xs], style,
                        color=COLOURS.get(grid, "0.4"), lw=1.7, marker="o", ms=4,
                        label=f"$N$={grid}, {engine}")
        ax.axhline(1.0, color="0.3", lw=1.2, zorder=1)
        ax.set_yscale("log")
        ax.set_xlabel("num_modes / $N$")
        ax.set_ylabel("cost relative to the plain real/imag transform")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, ncol=2, framealpha=.92, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
