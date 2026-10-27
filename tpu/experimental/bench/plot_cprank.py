"""Plots the CP rank sweep: accuracy against rank, with the device time it costs.

The point of the figure is the gap between the two curves.  Accuracy falls steeply with rank
while device time stays almost flat, because the rank is the contracted axis of the second
product and is narrower than the matrix unit until it approaches the channel count.

One panel per dataset, since the landmarks move with the channel count: the arithmetic ratio
against the dense filter is 2R/C, so the dense cost is predicted at R = C/2.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACC, TIME, REF = "#2f6fb2", "#c4622d", "#888888"
OUT = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/cprank/cprank.png"
# (panel title, channels, csv) -- the first three share a filter shape, so their rank ladders
# are the same and only the problem differs; Navier--Stokes is C = 64 and its landmarks move.
R = "tpu/results/cprank2/tab_cprank_"
PANELS = [("Burgers", 24, R + "Burgers.csv"),
          ("Darcy $32^2$", 24, R + "Darcy-32.csv"),
          ("Darcy $64^2$", 24, R + "Darcy-64.csv"),
          ("Navier--Stokes $128^2$", 64, R + "NS-128.csv")]


def load(path):
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok"]
    ranks = sorted(int(r["rank"]) for r in rows if r["filter"] == "tfno")
    by = {int(r["rank"]): r for r in rows if r["filter"] == "tfno"}
    flat = {r["filter"]: r for r in rows if r["filter"] in ("dense", "f-fno")}
    return ranks, by, flat


def main():
    panels = [(t, c, load(p)) for t, c, p in PANELS]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.0))
    axes = [axes] if len(panels) == 1 else list(axes)

    for ax, (title, channels, (ranks, by, flat)) in zip(axes, panels):
        h1 = [float(by[r]["h1"]) for r in ranks]
        us = [float(by[r]["us_mean"]) for r in ranks]

        ax.plot(ranks, h1, "o-", color=ACC, lw=2, ms=6, label="TFNO, rel $H^1$", zorder=3)
        ax.set_xlabel("CP rank $R$")
        ax.set_ylabel("test rel $H^1$", color=ACC)
        ax.tick_params(axis="y", labelcolor=ACC)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ranks)
        ax.set_xticklabels([str(r) for r in ranks], fontsize=8)
        ax.grid(alpha=.25)

        # The filters the sweep is placed between, at the same batch on the same weights.
        # Their errors can sit within a hair of each other, so the labels are staggered
        # to opposite sides of their lines rather than overlapping each other.
        for name, style, label, dy in (("dense", (0, (4, 2)), "dense", 3),
                                       ("f-fno", (0, (1, 2)), "F-FNO", -11)):
            if name in flat:
                y = float(flat[name]["h1"])
                ax.axhline(y, color=ACC, ls=style, lw=1.3, alpha=.75)
                ax.annotate(label, (ranks[0], y), ha="left",
                            textcoords="offset points", xytext=(3, dy),
                            fontsize=8, color=ACC)

        # where the arithmetic says the CP form should already cost what the dense filter costs
        half = channels // 2
        if ranks[0] <= half <= ranks[-1]:
            ax.axvline(half, color=REF, ls=":", lw=1)
            ax.annotate(f"$R=C/2$", (half, max(h1)), ha="center",
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=8, color="#555555")

        ax2 = ax.twinx()
        ax2.plot(ranks, us, "s--", color=TIME, lw=2, ms=5, label="device time")
        ax2.set_ylabel("device time ($\\mu$s / sample)", color=TIME)
        ax2.tick_params(axis="y", labelcolor=TIME)
        # a flat line should look flat, so the axis starts at zero rather than at the data
        ax2.set_ylim(0, max(us) * 1.35)

        ax.set_title(f"{title}  ($C = {channels}$, 300 epochs)", fontsize=10)
        h_a, l_a = ax.get_legend_handles_labels()
        h_b, l_b = ax2.get_legend_handles_labels()
        ax.legend(h_a + h_b, l_a + l_b, loc="upper right", fontsize=8, framealpha=.9)

    fig.suptitle("CP rank against accuracy and device time, precision highest, v5e", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
