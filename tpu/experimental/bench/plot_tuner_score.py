"""What the predicted batch costs, for every configuration the tuner was scored on.

The batch sweeps show that the minimum is a choice; these panels show whether the tuner makes
it. Each row is one configuration and each marker is a batch priced against the cheapest batch
that configuration's sweep found, so one is the sweep's own floor and the distance to the right
is what the choice costs.

Two markers per row, because the closed form and the ladder it seeds are worth scoring apart.
The form stops at the grain, or at capacity when capacity is below the grain, and the optimum
often keeps improving past that point -- so it reads low wherever capacity is large, and that
is most of the error on both panels.

The shaded band is the reason the right panel is a cost and not a hit rate. It is how far the
two independent GPU sweeps disagree with each other about the fastest batch: on that hardware
the curve is a smooth bowl with no grain, the floor is flat, and the argmin is not a stable
quantity. Anything inside the band is inside the measurement, so landing there is as close as
the experiment can resolve. The TPU rows have no band because the grain makes the curve a
sawtooth, where a batch between two multiples loses to both and the argmin is sharp.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TPU = "tpu/results/tuner_scorecard_tpu.csv"
GPU = "tpu/results/gpu5070/tuner_scorecard_gpu.csv"
REPRO = "tpu/results/gpu5070/tuner_reproducibility_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/tuner_scorecard.png"
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
SHORT = {"float32": "f32", "bfloat16": "bf16"}
# The two GPU sweeps are separate scoring targets, not one: they were run in different sessions
# on different ladders, and the whole point of the band is that they do not agree.
SWEEP_TAG = {"gpu_sweep.csv": "5070 A", "sweep_pair.csv": "5070 B"}
FORM, PICK = "#c4622d", "#2f6fb2"
BAND = "#3f8f52"


def rows_of(path, tag=None):
    if not os.path.exists(path):
        raise SystemExit(f"no scorecard at {path}; run the scorer first")
    out = []
    for row in csv.DictReader(open(path)):
        label = tag(row) if tag else row["target"].replace("TPU ", "")
        out.append({
            "block": label,
            "name": f"{row['case']} {SHORT.get(row['dtype'], row['dtype'])}",
            "case": row["case"], "dtype": row["dtype"],
            "form": float(row["cost_form"]) if row["cost_form"] else None,
            "pick": float(row["cost_pick"]) if row["cost_pick"] else None,
        })
    return out


def ordered(rows):
    """Grouped by target, then by dataset and precision, so blocks read the same way."""
    blocks = collections.defaultdict(list)
    for row in rows:
        blocks[row["block"]].append(row)
    out = []
    for block in sorted(blocks, key=lambda b: (b.startswith("5070"), b)):
        out.append((block, sorted(blocks[block],
                                  key=lambda r: (ORDER.index(r["case"]), r["dtype"]))))
    return out


def band_extent(blocks):
    """Row span of the GPU blocks, which are the only ones the band applies to."""
    y, lo, hi = 0, None, None
    for block, group in blocks:
        for _ in group:
            if block.startswith("5070"):
                lo = y if lo is None else lo
                hi = y
            y += 1
    return lo, hi


def detail(ax, blocks, repro_worst):
    labels, y = [], 0
    worst_form = worst_pick = (1.0, 0, "")
    lo, hi = band_extent(blocks)
    if lo is not None:
        ax.fill_betweenx([lo - .5, hi + .5], 1.0, repro_worst, color=BAND, alpha=.13,
                         lw=0, zorder=0)
    for block, group in blocks:
        if y:
            ax.axhline(y - .5, color="0.8", lw=.8, zorder=1)
        for row in group:
            labels.append(f"{block}  {row['name']}")
            form, pick = row["form"], row["pick"]
            if form is not None and pick is not None:
                ax.plot([min(form, pick), max(form, pick)], [y, y], "-", color="0.75",
                        lw=1.0, zorder=2)
            if form is None:
                # The form lands on a batch the sweep never measured, so it has no price. The
                # square says so in place of a dot rather than leaving the row looking perfect.
                ax.plot(1.205, y, "s", mfc="none", mec="0.55", ms=5, mew=1.1, zorder=3)
            else:
                ax.plot(form, y, "o", mfc="none", mec=FORM, ms=7, mew=1.7, zorder=4)
                if form > worst_form[0]:
                    worst_form = (form, y, f"{form:.3f}$\\times$")
            if pick is not None:
                ax.plot(pick, y, "o", color=PICK, ms=6.5, zorder=5)
                if pick > worst_pick[0]:
                    worst_pick = (pick, y, f"{pick:.3f}$\\times$")
            y += 1

    for cost, row, text in (worst_form, worst_pick):
        if cost > 1.0:
            ax.annotate(text, (cost, row), textcoords="offset points", xytext=(7, -3),
                        fontsize=7.5, color=FORM if cost == worst_form[0] else PICK)
    ax.axvline(1.0, color="0.3", lw=.9, zorder=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5, fontfamily="monospace")
    ax.set_ylim(len(labels) - .5, -.5)
    ax.set_xlim(.995, 1.225)
    ax.set_xlabel("cost relative to the cheapest batch this sweep measured")
    ax.set_title("every configuration", fontsize=11)
    ax.grid(axis="x", alpha=.2)
    handles = [
        plt.Line2D([], [], ls="none", marker="o", mfc="none", mec=FORM, ms=7, mew=1.7,
                   label="closed form $B^*$ alone"),
        plt.Line2D([], [], ls="none", marker="o", color=PICK, ms=6.5,
                   label="the batch the candidate ladder picks"),
        plt.Line2D([], [], ls="none", marker="s", mfc="none", mec="0.55", ms=5, mew=1.1,
                   label="form's batch not on the swept grid"),
        plt.Line2D([], [], lw=6, color=BAND, alpha=.3,
                   label=f"GPU sweep-to-sweep spread, to {repro_worst:.3f}$\\times$"),
    ]
    ax.legend(handles=handles, fontsize=7.5, framealpha=.92, loc="lower right")


def summary(ax, rows, repro_worst):
    """How many configurations come in under a given cost -- the whole scorecard in one view."""
    for key, colour, name in ((("form"), FORM, "closed form $B^*$ alone"),
                              (("pick"), PICK, "candidate ladder's pick")):
        costs = sorted(r[key] for r in rows if r[key] is not None)
        if not costs:
            continue
        xs, ys = [1.0], [0.0]
        for i, c in enumerate(costs, start=1):
            xs += [c, c]
            ys += [ys[-1], i / len(costs)]
        xs.append(1.225)
        ys.append(ys[-1])
        ax.plot(xs, ys, "-", color=colour, lw=2.0,
                label=f"{name}  ($n$={len(costs)}, worst {costs[-1]:.3f}$\\times$)")
    # The band is measured on the GPU rows only, but it belongs on this panel too: it is the
    # finest distinction the experiment can draw anywhere, and so the scale the curves are read
    # against rather than a property of one target.
    ax.axvspan(1.0, repro_worst, color=BAND, alpha=.13, lw=0, zorder=0,
               label=f"GPU sweep-to-sweep spread, to {repro_worst:.3f}$\\times$")
    ax.axvline(1.0, color="0.3", lw=.9, zorder=1)
    ax.set_xlim(.995, 1.225)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("cost relative to the cheapest batch measured")
    ax.set_ylabel("fraction of configurations at or under")
    ax.set_title("what the choice costs", fontsize=11)
    ax.grid(alpha=.2)
    ax.legend(fontsize=7.5, framealpha=.92, loc="lower right")


def main():
    rows = rows_of(TPU) + rows_of(GPU, tag=lambda r: SWEEP_TAG[r["sweep"]])
    repro = list(csv.DictReader(open(REPRO)))
    repro_worst = max(float(r[k]) for r in repro
                      for k in ("cost_b_on_a", "cost_a_on_b") if r[k])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 8.0),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    detail(axes[0], ordered(rows), repro_worst)
    summary(axes[1], rows, repro_worst)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
