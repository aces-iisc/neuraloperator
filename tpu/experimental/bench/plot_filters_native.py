"""What the factorised filters buy, and what our transform buys underneath each of them.

The left panel holds the transform fixed and varies the filter, which is the factorisation's own
contribution. The right panel holds the filter fixed and varies the transform, which is ours.
Separating them is the point: a single end-to-end number cannot say which of the two a speed-up
came from, and the two do not compose the way a reader would assume.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "tpu/results/gpu5070/filters_vs_native_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/gpu_filters_native.png"
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
FILTERS = ["dense", "ffno", "tfno"]
COLOURS = {"Burgers": "#2f6fb2", "Darcy-32": "#c4622d",
           "Darcy-64": "#3f8f52", "NS-128": "#8a5bb5"}


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no sweep at {path}; run filters_vs_native_gpu on the GPU first")
    return {(r["case"], r["filter"], r["impl"]): float(r["us_per_sample"])
            for r in csv.DictReader(open(path))}


def bars(ax, at, series, title, ylabel):
    width = 0.8 / len(ORDER)
    for i, case in enumerate(ORDER):
        xs, ys = [], []
        for j, which in enumerate(FILTERS):
            value = series(at, case, which)
            if value is None:
                continue
            xs.append(j + (i - (len(ORDER) - 1) / 2) * width)
            ys.append(value)
        ax.bar(xs, ys, width=width * 0.92, color=COLOURS[case], label=case)
    ax.axhline(1.0, color="0.3", lw=1.0, zorder=3)
    ax.set_xticks(range(len(FILTERS)))
    ax.set_xticklabels(FILTERS)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8, framealpha=.92)


def main():
    at = load(SWEEP)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    bars(axes[0], at,
         lambda a, c, f: (a[(c, "dense", "ours")] / a[(c, f, "ours")]
                          if (c, f, "ours") in a and (c, "dense", "ours") in a else None),
         "what the filter buys, at our transform",
         "speed-up over the dense filter")
    bars(axes[1], at,
         lambda a, c, f: (a[(c, f, "native")] / a[(c, f, "ours")]
                          if (c, f, "native") in a and (c, f, "ours") in a else None),
         "what our transform buys, at each filter",
         "native torch.fft time / ours")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
