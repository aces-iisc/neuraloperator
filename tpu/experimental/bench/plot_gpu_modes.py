"""Cost against the kept band on the GPU, for both torch spellings.

The left panel is what a mode sweep is usually for: how much a wider band costs. The right panel
is the one that carries an argument. Our transform computes only the modes it keeps and the
textbook spelling computes the whole spectrum and throws most of it away, so the partial DFT
should pay off most at a narrow band and least at a wide one. On the three small grids it does,
and the gap still never closes -- even at the narrowest band we are behind. On Navier--Stokes the
trend reverses, because at `channels` of 64 the channel matmul rather than the transform sets the
cost, and that is the one shape whose arithmetic suits the hardware.

Accuracy is not swept here. What a band is worth in error is a property of the problem and the
trained weights, not of the device, so it is measured once on the TPU and not repeated.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "tpu/results/gpu5070/mode_sweep_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/gpu_mode_sweep.png"
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
COLOURS = {"Burgers": "#2f6fb2", "Darcy-32": "#c4622d",
           "Darcy-64": "#3f8f52", "NS-128": "#8a5bb5"}
# The band each dataset was trained at, and so the one every other table reports.
TRAINED = {"Burgers": 16, "Darcy-32": 16, "Darcy-64": 16, "NS-128": 64}


def load(path):
    series = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        series[row["case"]][int(row["n_modes"])] = (
            float(row["ours_us_per_sample"]), float(row["native_us_per_sample"]))
    return series


def main():
    if not os.path.exists(SWEEP):
        raise SystemExit(f"no sweep at {SWEEP}; run mode_sweep_gpu on the GPU first")
    series = load(SWEEP)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for case in ORDER:
        data = series.get(case)
        if not data:
            continue
        xs = sorted(data)
        colour = COLOURS[case]
        axes[0].plot(xs, [data[x][0] for x in xs], "o-", color=colour, lw=1.8, ms=4,
                     label=f"{case}, ours")
        axes[0].plot(xs, [data[x][1] for x in xs], "o--", color=colour, lw=1.3, ms=3,
                     alpha=.65, label=f"{case}, native")
        axes[1].plot(xs, [data[x][0] / data[x][1] for x in xs], "o-", color=colour,
                     lw=1.8, ms=4, label=case)
        trained = TRAINED[case]
        if trained in data:
            axes[1].plot(trained, data[trained][0] / data[trained][1], "*", color=colour,
                         ms=15, zorder=5)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("device time ($\\mu$s / sample)")
    axes[0].set_title("cost of the band", fontsize=11)
    axes[0].legend(fontsize=6.5, ncol=2, framealpha=.92, loc="upper left")

    axes[1].axhline(1.0, color="0.3", lw=.9, zorder=0)
    axes[1].set_ylabel("ours / native torch")
    axes[1].set_title("what the partial transform is worth", fontsize=11)
    axes[1].legend(fontsize=8, framealpha=.92, loc="upper left")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("num_modes")
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"], fontsize=8)
        ax.set_xticks([], minor=True)
        ax.grid(alpha=.2, which="major")

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
