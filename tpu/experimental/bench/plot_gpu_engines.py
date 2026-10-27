"""Which real-input transform to use on a GPU, against the two things that decide it.

Both panels are priced against the plain real/imag partial transform, so one is that transform
and a point below one is a packing scheme that beat it. The two panels hold one axis fixed and
sweep the other, because the schemes trade along different ones: coupled pairs the signals, so
what it changes is the operand's width; reduced splits the samples, so what it changes is the
contracted length. Neither trade can be judged from a single shape, which is what the earlier
TPU study measured.

Reduced is absent on Burgers. The dataset's row axis is 17 samples long, an even/odd split needs
an even length, and no amount of padding makes that free. Being inapplicable is the answer for
that occasion, so the line is missing rather than imputed.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = "tpu/results/gpu5070/engines_gpu.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/gpu_engines.png"
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
COLOURS = {"Burgers": "#2f6fb2", "Darcy-32": "#c4622d",
           "Darcy-64": "#3f8f52", "NS-128": "#8a5bb5"}
STYLE = {"coupled": "-", "reduced": "--"}
# The band each dataset is trained at, and the width a layer actually presents at a tuned batch.
TRAINED = {"Burgers": 16, "Darcy-32": 16, "Darcy-64": 16, "NS-128": 64}
WIDE = 16384


def load(path):
    """{(case, band, width, engine): us}, plus the widths and bands measured."""
    at, widths, bands = {}, set(), collections.defaultdict(set)
    if not os.path.exists(path):
        raise SystemExit(f"no engine sweep at {path}; run engines_gpu on the GPU first")
    for row in csv.DictReader(open(path)):
        if row["status"] != "ok":
            continue
        case, m, k = row["case"], int(row["n_modes"]), int(row["k_cols"])
        at[(case, m, k, row["engine"])] = float(row["us"])
        widths.add(k)
        bands[case].add(m)
    return at, sorted(widths), bands


def ratios(at, case, engine, fixed, sweep_values, by_band):
    """The scheme's cost over reim, along whichever axis the panel sweeps."""
    xs, ys = [], []
    for v in sweep_values:
        key = (case, v, fixed, engine) if by_band else (case, fixed, v, engine)
        base = (case, v, fixed, "reim") if by_band else (case, fixed, v, "reim")
        if key in at and base in at:
            xs.append(v)
            ys.append(at[key] / at[base])
    return xs, ys


def panel(ax, at, cases, sweep_of, fixed_of, by_band, title, xlabel):
    for case in ORDER:
        if case not in cases:
            continue
        for engine, style in STYLE.items():
            xs, ys = ratios(at, case, engine, fixed_of(case), sweep_of(case), by_band)
            if not xs:
                continue
            ax.plot(xs, ys, style, color=COLOURS[case], lw=1.7, ms=4, marker="o",
                    label=f"{case}, {engine}")
    ax.axhline(1.0, color="0.3", lw=1.2, zorder=1)
    ax.set_xscale("log", base=2)
    # Log, because the spread runs from twice to ten times and the thing worth seeing is that
    # the floor is never approached: on a linear axis the gap to one reads as a margin rather
    # than as the whole answer.
    ax.set_yscale("log")
    ax.set_ylim(0.9, None)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cost relative to the plain real/imag transform")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=.2, which="major")
    ax.legend(fontsize=6.8, ncol=2, framealpha=.92, loc="upper left")


def main():
    at, widths, bands = load(SWEEP)
    wide = WIDE if WIDE in widths else max(widths)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    panel(axes[0], at, bands, lambda c: sorted(bands[c]), lambda c: wide, True,
          f"against num_modes, at {wide} signals", "num_modes")
    panel(axes[1], at, bands, lambda c: widths, lambda c: TRAINED[c], False,
          "against the width, at each dataset's trained num_modes", "signals transformed at once")
    for ax, ticks in ((axes[0], [2, 4, 8, 16, 32, 64, 128]), (axes[1], widths)):
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
        ax.set_xticks([], minor=True)

    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
