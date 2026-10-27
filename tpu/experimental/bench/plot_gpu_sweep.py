"""Device time per sample against batch on the GPU, one panel per torch spelling.

The companion to the TPU panels, and the reason the batch is measured rather than assumed.
Two things differ from the TPU plot and both are findings rather than presentation choices.

There is no batch grain here, so there are no grain verticals: the curves are smooth bowls,
not sawtooths, because nothing in the hardware consumes the batch in fixed-width passes. What
sets the minimum instead is the working set, so each panel marks the predicted capacity for
each dataset and the batch the ladder actually chose.

Both spellings share the axes, so the offset between the panels is the cost of our transform
on this hardware, and the shape within each panel is what the batch is worth.
"""
import collections
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "tpu/results/gpu5070/sweep_pair.csv"
TUNED = "tpu/results/gpu5070/tune_live_gpu.csv"
# Absolute time spans two and a half decades across the four datasets, which flattens every
# curve into a line.  Dividing each by its own fastest batch puts all four on one linear scale,
# where what the panel is about -- how sharp the optimum is, and what a batch off it costs --
# becomes readable.
NORM = os.environ.get("GPU_SWEEP_NORM") == "1"
OUT = (sys.argv[1] if len(sys.argv) > 1 else
       f"tpu/results/gpu_sweep_pair{'_relative' if NORM else ''}.png")
PANELS = ["ours torch", "native torch"]
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
COLOURS = {"Burgers": "#2f6fb2", "Darcy-32": "#c4622d",
           "Darcy-64": "#3f8f52", "NS-128": "#8a5bb5"}


def load(path):
    """{impl: {case: {batch: us}}} plus {(impl, case): (capacity, ladder pick)}."""
    series = collections.defaultdict(lambda: collections.defaultdict(dict))
    marks = {}
    for row in csv.DictReader(open(path)):
        impl, case, batch = row["impl"], row["case"], int(row["batch"])
        us, cap = float(row["us_per_sample"]), int(row["capacity"])
        series[impl][case][batch] = us
        key = (impl, case)
        marks.setdefault(key, (cap, None, float("inf")))
        # The ladder picks the cheapest of the candidates it offered, so keep the running best.
        if int(row["is_candidate"]) and us < marks[key][2]:
            marks[key] = (cap, batch, us)
    return series, marks


def tuner_batches():
    """{case: batch} the live tuner returned at the shipped band, float32: its real answer.

    Read from the end-to-end run rather than recomputed from the candidate column here. The
    ladder is a step inside the search, not the search's output, and marking it would put a
    batch on the panel that the tuner never returns.
    """
    if not os.path.exists(TUNED):
        return {}
    return {r["case"]: int(r["b_best"]) for r in csv.DictReader(open(TUNED))
            if r["dtype"] == "float32"}


def on_curve(xs, ys, target):
    """The curve's height at `target`, interpolated between its neighbours.

    The tuner refines between the sweep's points, so its batch usually is not one of them and
    has no measured height here. Interpolating places the marker at the right position along
    the axis that carries the claim -- which batch was chosen -- without inventing a timing:
    the sweep and the tuner ran in different sessions, so its own measurement is not on this
    curve's scale anyway.
    """
    if target <= xs[0]:
        return ys[0]
    if target >= xs[-1]:
        return ys[-1]
    i = max(i for i, x in enumerate(xs) if x <= target)
    if xs[i] == target:
        return ys[i]
    lo, hi = xs[i], xs[i + 1]
    frac = (math.log(target) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return ys[i] + frac * (ys[i + 1] - ys[i])


def main():
    series, marks = load(SWEEP)
    chosen = tuner_batches()
    present = [p for p in PANELS if series.get(p)]
    if not present:
        raise SystemExit(f"no sweep rows in {SWEEP}; run sweep_pair on the GPU first")

    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 4.4),
                             sharey=True)
    axes = [axes] if len(present) == 1 else list(axes)
    ticks = set()
    for ax, impl in zip(axes, present):
        for case in ORDER:
            data = series[impl].get(case)
            if not data:
                continue
            xs = sorted(data)
            ys = [data[x] for x in xs]
            floor = min(ys)
            if NORM:
                ys = [y / floor for y in ys]
            ticks.update(xs)
            best = min(range(len(ys)), key=lambda i: ys[i])
            cap, _, _ = marks[(impl, case)]
            tuned = chosen.get(case) if impl == "ours torch" else None
            # The optimum goes in the legend rather than beside the star: the four curves
            # bottom out at nearly the same height once normalised, and four labels overprint.
            label = f"{case}   best $B$={xs[best]}, {floor:.3g} $\\mu$s/sa."
            ax.plot(xs, ys, "o-", color=COLOURS[case], lw=1.8, ms=4.5, label=label)
            # One convention across every GPU panel: star is what the tuner chose, ring is
            # what the sweep found fastest.
            if tuned is not None:
                ax.plot(tuned, on_curve(xs, ys, tuned), "*", color=COLOURS[case], ms=17,
                        zorder=6)
            ax.plot(xs[best], ys[best], "o", mfc="none", mec=COLOURS[case], ms=12, mew=1.6,
                    zorder=4)
            ax.axvline(cap, color=COLOURS[case], ls=":", lw=1.0, alpha=.65, zorder=0)
        ax.set_xscale("log", base=2)
        if NORM:
            ax.axhline(1.0, color="0.3", lw=.8, zorder=0)
            # A batch of one costs over a hundred times the optimum, which on a shared linear
            # axis compresses every curve near its minimum into the same flat line.  The claim
            # is about the shape of the bowl, so the view stops just above twice the optimum
            # and the starved left tail runs off the top.
            ax.set_ylim(0.93, 2.3)
        else:
            ax.set_yscale("log")
        ax.set_xlabel("batch")
        ax.set_title(impl, fontsize=11)
        ax.grid(alpha=.2, which="major")
    axes[0].set_ylabel("cost relative to this dataset's fastest batch" if NORM
                      else "device time ($\\mu$s / sample)")
    for ax in axes:
        ax.legend(fontsize=7.5, framealpha=.92, loc="upper right")

    # The sweep places points a few percent apart near the knee -- 127 beside 140 beside 170 --
    # so labelling every measured batch overprints and reads as a smear.  A decade ladder plus
    # the four capacities carries the argument; everything else is plotted, just unlabelled.
    caps = {cap for cap, _, _ in marks.values()}
    shown = sorted(caps)
    # The capacities carry the argument, so they are placed first and a decade tick is only
    # added where it will not collide with one on a log axis.
    for t in sorted({1, 2, 4, 8, 16, 32, 64, 128, 256, 512}):
        if t <= max(ticks) and all(abs(t / s - 1) > 0.2 for s in shown):
            shown.append(t)
    shown = sorted(shown)
    for ax in axes:
        ax.set_xticks(shown)
        ax.set_xticklabels([str(t) for t in shown], fontsize=7)
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="x", pad=2)

    headline = ("Batch sweep relative to each dataset's own best" if NORM
                else "Batch sweep, device time per sample")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
