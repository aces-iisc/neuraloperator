"""What num_modes is worth against what it costs, on a GPU.

The left panel is the only measurement here that needs training: every point is its own run, so
what a wider num_modes buys in accuracy can be read off directly. The right panel prices the same
values on the device. Each curve is relative to its own dataset's best, because the four differ
by two orders of magnitude in both error and time and a shared absolute axis flattens all four
into lines.

The marked value is the smallest whose error the seed spread cannot separate from the best,
rather than the error's argmin; past that point the curve is flat to within what a rerun
reproduces, so the lowest point out there is noise with a price tag. Both are drawn.
"""
import collections
import csv
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCURACY = "tpu/results/gpu5070/mode_accuracy_gpu.csv"
COST = "tpu/results/gpu5070/mode_sweep_gpu.csv"
BATCH = "tpu/results/gpu5070/batch_at_mode_gpu.csv"
NOISE = "tpu/results/gpu5070/mode_noise_gpu.csv"
TUNED = "tpu/results/gpu5070/tune_live_knee_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/gpu_mode_to_batch.png"
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
COLOURS = {"Burgers": "#2f6fb2", "Darcy-32": "#c4622d",
           "Darcy-64": "#3f8f52", "NS-128": "#8a5bb5"}
FALLBACK_TOLERANCE = 1.01


def need(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run the GPU scripts first")
    return path


def tolerance():
    """Each dataset's own seed-to-seed spread. Differences inside it are not differences.

    The spreads are not alike across datasets, so one number for all of them would be wrong in
    both directions at once. A dataset with no seed study takes the median of those measured:
    the widest belongs to the smallest training set, which is why it scatters, and lending that
    to the largest dataset in the study would borrow from the one least like it.
    """
    if not os.path.exists(NOISE):
        return collections.defaultdict(lambda: FALLBACK_TOLERANCE)
    groups = collections.defaultdict(list)
    for row in csv.DictReader(open(NOISE)):
        groups[(row["case"], row["n_modes"])].append(float(row["rel_h1"]))
    per_case = collections.defaultdict(list)
    for (case, _), errors in groups.items():
        if len(errors) > 1:
            per_case[case].append(max(errors) / min(errors))
    spread = {case: max(v) for case, v in per_case.items()}
    fallback = statistics.median(spread.values()) if spread else FALLBACK_TOLERANCE
    return collections.defaultdict(lambda: fallback, spread)


def accuracy(tol):
    """{case: {band: h1}}, and the knee and argmin each case selects."""
    errors = collections.defaultdict(dict)
    for row in csv.DictReader(open(need(ACCURACY))):
        errors[row["case"]][int(row["n_modes"])] = float(row["rel_h1"])
    chosen = {}
    for case, curve in errors.items():
        best = min(curve, key=curve.get)
        knee = min(b for b in curve if curve[b] <= tol[case] * curve[best])
        chosen[case] = (knee, best)
    return errors, chosen


def cost():
    series = collections.defaultdict(dict)
    for row in csv.DictReader(open(need(COST))):
        series[row["case"]][int(row["n_modes"])] = float(row["ours_us_per_sample"])
    return series


def batches():
    """{case: (band, {batch: us}, {candidate batches})}."""
    series, cands, band = collections.defaultdict(dict), collections.defaultdict(set), {}
    for row in csv.DictReader(open(need(BATCH))):
        case, b = row["case"], int(row["batch"])
        series[case][b] = float(row["us_per_sample"])
        band[case] = int(row["n_modes"])
        if int(row["is_candidate"]):
            cands[case].add(b)
    return {c: (band[c], series[c], cands[c]) for c in series}


def relative(ax, curve, colour, label):
    xs = sorted(curve)
    floor = min(curve.values())
    ax.plot(xs, [curve[x] / floor for x in xs], "o-", color=colour, lw=1.8, ms=4, label=label)
    return floor


def panel_accuracy(ax, errors, chosen, tol):
    for case in ORDER:
        curve = errors.get(case)
        if not curve:
            continue
        colour = COLOURS[case]
        knee, best = chosen[case]
        floor = relative(ax, curve, colour,
                         f"{case}   num_modes={knee}, $H^1$ {curve[knee]:.4f}, "
                         f"seeds $\\pm${100 * (tol[case] - 1):.0f}%")
        # Everything inside this dataset's own band is one training run's worth of luck, so the
        # curve stops carrying information once it enters -- which is what makes the knee a
        # defensible choice and the argmin out on the flat not one.
        ax.axhspan(1.0, tol[case], color=colour, alpha=.07, lw=0, zorder=0)
        ax.plot(knee, curve[knee] / floor, "*", color=colour, ms=17, zorder=5)
        if best != knee:
            ax.plot(best, curve[best] / floor, "o", mfc="none", mec=colour, ms=12, mew=1.6,
                    zorder=4)
    ax.set_ylim(0.97, 1.9)
    ax.axhline(1.0, color="0.3", lw=.8, zorder=1)
    ax.set_ylabel("test $H^1$ relative to this dataset's best num_modes")
    ax.set_title("what num_modes is worth", fontsize=11)
    ax.legend(fontsize=7, framealpha=.92, loc="upper right")


def panel_cost(ax, series, chosen):
    for case in ORDER:
        curve = series.get(case)
        if not curve:
            continue
        colour = COLOURS[case]
        floor = relative(ax, curve, colour, case)
        knee = chosen.get(case, (None, None))[0]
        if knee in curve:
            ax.plot(knee, curve[knee] / floor, "*", color=colour, ms=16, zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel("device time relative to this dataset's narrowest num_modes")
    ax.set_title("what num_modes costs", fontsize=11)
    ax.legend(fontsize=8, framealpha=.92, loc="upper left")


def tuner_batches():
    """{case: batch} the live tuner returned at the knee band -- its actual answer."""
    if not os.path.exists(TUNED):
        return {}
    return {r["case"]: int(r["b_best"]) for r in csv.DictReader(open(TUNED))
            if r["dtype"] == "float32"}


def panel_batch(ax, data, chosen_batch):
    for case in ORDER:
        if case not in data:
            continue
        band, curve, _ = data[case]
        colour = COLOURS[case]
        xs = sorted(curve)
        floor = min(curve.values())
        best = min(curve, key=curve.get)
        tuned = chosen_batch.get(case)
        ax.plot(xs, [curve[x] / floor for x in xs], "o-", color=colour, lw=1.8, ms=4,
                label=f"{case} at num_modes={band}   best $B$={best}, {floor:.3g} $\\mu$s/sa.")
        # Same convention as the first panel: star is the tuner's answer, ring is the winner.
        if tuned is not None:
            ax.plot(tuned, curve.get(tuned, floor) / floor, "*", color=colour, ms=17, zorder=5)
        ax.plot(best, 1.0, "o", mfc="none", mec=colour, ms=12, mew=1.6, zorder=4)
    ax.set_xscale("log", base=2)
    # A batch of one costs over a hundred times the optimum, which on a shared linear axis
    # compresses every curve near its minimum into the same flat line. The claim is about the
    # shape of the bowl, so the view stops just above twice the optimum.
    ax.set_ylim(0.95, 2.3)
    ax.axhline(1.0, color="0.3", lw=.8, zorder=0)
    ax.set_xlabel("batch")
    ax.set_ylabel("device time relative to this dataset's best batch")
    ax.set_title("the batch, at the chosen num_modes", fontsize=11)
    ax.legend(fontsize=7, framealpha=.92, loc="upper right")


def main():
    tol = tolerance()
    errors, chosen = accuracy(tol)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9))
    panel_accuracy(axes[0], errors, chosen, tol)
    panel_cost(axes[1], cost(), chosen)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("num_modes")
        ax.set_xticks([2, 4, 8, 16, 32, 64, 128])
        ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"], fontsize=8)
        ax.set_xticks([], minor=True)
    for ax in axes:
        ax.grid(alpha=.2, which="major")
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
