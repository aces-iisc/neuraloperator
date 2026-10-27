"""The transform and the channel filter, changed one at a time.

Three bars per filter. The first is the published shape -- torch.fft and a complex einsum.
The second keeps torch.fft and swaps in our real block matmul, so the only thing that moved
is how the channels are mixed. The third moves the transform as well. The gap between the
first two is the filter's contribution and the gap between the last two is the transform's.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = "tpu/results/gpu5070/filters_threeway_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/filters_threeway.png"
CASES = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
FILTERS = [("dense", "dense"), ("ffno", "F-FNO"), ("tfno", "TFNO")]
ARMS = [("native+native", "torch.fft + published filter", "#7a7f87"),
        ("native+ours", "torch.fft + our filter", "#c4622d"),
        ("ours+ours", "our transform + our filter", "#2f6fb2")]


def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"no sweep at {path}; run filters_threeway_gpu on the GPU first")
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok"]
    at = {(r["case"], r["filter"], r["arm"]): float(r["us_per_sample"]) for r in rows}
    meta = {r["case"]: (int(r["grid"]), int(r["channels"]), int(r["n_modes"]), int(r["batch"]))
            for r in rows}
    return at, meta


def main():
    at, meta = load(SWEEP)
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.3))
    width = 0.26
    for ax, case in zip(axes, CASES):
        grid, channels, n_modes, batch = meta[case]
        for i, (arm, label, colour) in enumerate(ARMS):
            xs = [j + (i - 1) * width for j in range(len(FILTERS))]
            ys = [at.get((case, key, arm), np.nan) for key, _ in FILTERS]
            bars = ax.bar(xs, ys, width=width * 0.92, color=colour,
                          label=label if case == CASES[0] else None)
            ax.bar_label(bars, fmt="%.1f", fontsize=6.5, padding=1.5)
        ax.set_xticks(range(len(FILTERS)))
        ax.set_xticklabels([name for _, name in FILTERS], fontsize=9)
        ax.set_xlabel("channel filter", fontsize=9)
        ax.set_title(f"{case}\n$N={grid}$, $C={channels}$, "
                     f"$\\mathrm{{num\\_modes}}={n_modes}$, batch {batch}", fontsize=9)
        ax.grid(axis="y", alpha=.2)
        ax.set_ylim(0, max(at.get((case, k, a), 0) for k, _ in FILTERS
                           for a, _, _ in ARMS) * 1.18)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("device time per sample  ($\\mu$s)", fontsize=9)
    fig.legend(loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(.5, -.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=.22)
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
