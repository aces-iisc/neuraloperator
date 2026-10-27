"""Device time of the three real-input row transforms against the band they produce.

One panel per grid.  The point of the figure is that the three curves never cross: the ordering
the tuner assumes holds at every band, not just at the one point Table~\\ref{tab:coupled-partial}
measures.
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/engmodes/tab_enginemodes.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tpu/results/engmodes/enginemodes.png"
STYLE = [("reim", "#2f6fb2", "o-"), ("coupled", "#c4622d", "s--"), ("reduced", "#3f8f52", "^-.")]


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "ok"]
    cases = []
    for r in rows:
        if r["case"] not in cases:
            cases.append(r["case"])

    fig, axes = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 4.1))
    axes = [axes] if len(cases) == 1 else list(axes)
    for ax, case in zip(axes, cases):
        sub = [r for r in rows if r["case"] == case]
        grid = sub[0]["grid"]
        for name, colour, mark in STYLE:
            pts = sorted((int(r["n_modes"]), float(r["device_us"]))
                         for r in sub if r["engine"] == name)
            ax.plot([p for p, _ in pts], [u for _, u in pts], mark, color=colour,
                    lw=2, ms=5, label=name)
        ax.set_xlabel("num_modes")   # plain text: mathtext would subscript the underscore
        ax.set_ylabel("device time ($\\mu$s per call)")
        ax.set_ylim(0, None)          # a constant ratio should look constant
        ax.grid(alpha=.25)
        ax.set_title(f"{case}  (grid {grid}, $K = {sub[0]['k_cols']}$)", fontsize=10)
        ax.legend(fontsize=9, framealpha=.9)

    fig.suptitle("The three real-input row transforms against the band, device time, v5e",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
