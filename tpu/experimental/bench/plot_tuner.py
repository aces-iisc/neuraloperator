"""Device time per sample against batch, one panel per accelerator, one line per dataset.

The panels are the argument for measuring the batch rather than deriving it.  Per-sample cost
falls while the free axis of the channel matmul is starved and rises again once the working set
stops fitting, so the minimum is a choice; and the minima sit on multiples of the batch grain,
which the dotted verticals mark.  A batch between two multiples loses to both, so the dips are
narrow and a grid of powers of two steps over some of them -- 384 is the fastest batch measured
for three of the v5e configurations and is not a power of two.

Both panels share the axes, so the generational shift is the offset between them.
"""
import collections
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [("TPU v5e", "tpu/results/tuner_v5e/tab_tuner.csv"),
        ("TPU v6e", "tpu/results/tuner_v6e/tab_tuner.csv")]
DTYPE = os.environ.get("TUNER_DTYPE", "float32")
# Absolute time spans two and a half decades across the four datasets, which flattens every
# curve into a line.  Dividing each by its own fastest batch puts all four on one linear scale,
# where what the panel is actually about -- how sharp the optimum is, and how much a batch off
# the grain costs -- becomes readable.
NORM = os.environ.get("TUNER_NORM") == "1"
OUT = (sys.argv[1] if len(sys.argv) > 1 else
       f"tpu/results/tuner_sweep_{DTYPE}{'_relative' if NORM else ''}.png")
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
COLOURS = {"Burgers": "#2f6fb2", "Darcy-32": "#c4622d",
           "Darcy-64": "#3f8f52", "NS-128": "#8a5bb5"}
GRAIN = 128


def load(path):
    """{case: {batch: us_per_sample}} for the requested precision."""
    series = collections.defaultdict(dict)
    if not os.path.exists(path):
        return series
    for row in csv.DictReader(open(path)):
        if row.get("role") != "sweep" or row.get("status") != "ok":
            continue
        if row.get("dtype") != DTYPE:
            continue
        try:
            value = float(row["us_per_sample"])
        except (KeyError, ValueError):
            continue
        if value == value:                      # drop the rows whose trace would not parse
            series[row["case"]][int(row["batch"])] = value
    return series


def main():
    panels = [(name, load(path)) for name, path in RUNS]
    present = [p for p in panels if any(p[1].values())]
    if not present:
        raise SystemExit("no sweep rows found; run tab_tuner on each target first")

    fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.3),
                             sharey=True)
    axes = [axes] if len(present) == 1 else list(axes)
    ticks = set()
    for ax, (name, series) in zip(axes, present):
        for case in ORDER:
            data = series.get(case)
            if not data:
                continue
            xs = sorted(data)
            ys = [data[x] for x in xs]
            if NORM:
                floor = min(ys)
                ys = [y / floor for y in ys]
            ticks.update(xs)
            best = min(range(len(ys)), key=lambda i: ys[i])
            # The optimum goes in the legend rather than beside the star: four stars land near
            # the same height once the curves are normalised, and four labels there overprint.
            legend = f"{case}   $B^*$={xs[best]}, {min(data.values()):.3g} $\\mu$s/sa."
            ax.plot(xs, ys, "o-", color=COLOURS[case], lw=1.8, ms=4.5, label=legend)
            ax.plot(xs[best], ys[best], "*", color=COLOURS[case], ms=15, zorder=5)
        for k in range(1, 5):
            ax.axvline(k * GRAIN, color="0.55", ls=":", lw=.9, zorder=0)
        ax.set_xscale("log", base=2)
        if NORM:
            ax.axhline(1.0, color="0.3", lw=.8, zorder=0)
        else:
            ax.set_yscale("log")
        ax.set_xlabel("batch")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=.2, which="major")
    axes[0].set_ylabel("cost relative to this dataset's fastest batch" if NORM
                      else "device time ($\\mu$s / sample)")
    # Each panel carries its own, since the optimum is what differs between the two machines.
    for ax in axes:
        ax.legend(fontsize=7.5, framealpha=.92, loc="upper left")

    # Only two groups of batches get a label.  The axis is logarithmic and the sweep deliberately
    # places points a few percent apart -- 113 beside 128 beside 141, 384 beside 400 -- so
    # labelling every measured batch overprints and reads as a smear.  The multiples of the grain
    # carry the argument, and below half a grain the points are far enough apart in log space to
    # label individually.  Everything else is still plotted, just unlabelled.
    shown = [t for t in sorted(ticks) if t % GRAIN == 0 or t < GRAIN // 2]
    for ax in axes:
        ax.set_xticks(shown)
        ax.set_xticklabels([str(t) for t in shown], fontsize=7.5)
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="x", pad=2)

    headline = ("Batch sweep relative to each dataset's own best" if NORM
                else "Batch sweep, device time per sample")
    fig.suptitle(f"{headline} -- {DTYPE}, whole four-layer model, trained weights, "
                 f"traces.\nStars mark the fastest batch; dotted lines are multiples "
                 f"of {GRAIN}.", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
