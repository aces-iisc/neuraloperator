"""Plots the num_modes sweep. num_modes is the TOTAL band width, as in neuralop's SpectralConv."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/modes/modes.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tpu/results/modes/modes_sweep.png"

rows = list(csv.DictReader(open(CSV)))
cases = []
for r in rows:
    if r["case"] not in cases:
        cases.append(r["case"])

fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2))
if len(cases) == 1:
    axes = [axes]

for ax, case in zip(axes, cases):
    sub = sorted([r for r in rows if r["case"] == case], key=lambda r: int(r["n_modes"]))
    nm = [int(r["n_modes"]) for r in sub]
    l2 = [float(r["rel_l2"]) for r in sub]
    us = [float(r["infer_us"]) for r in sub]

    ax.plot(nm, l2, "o-", color="#2f6fb2", lw=2, ms=6, label="rel $L^2$")
    ax.set_xlabel("num_modes"); ax.set_ylabel("test rel $L^2$", color="#2f6fb2")
    ax.tick_params(axis="y", labelcolor="#2f6fb2")
    ax.set_xticks(nm); ax.grid(alpha=.25)

    b = min(range(len(l2)), key=lambda i: l2[i])
    ax.axvline(nm[b], color="#888", ls=":", lw=1)
    ax.annotate(f"best rel $L^2$\nnum_modes={nm[b]}", (nm[b], l2[b]), textcoords="offset points",
                xytext=(8, 14), fontsize=8, color="#444")

    ax2 = ax.twinx()
    ax2.plot(nm, us, "s--", color="#c4622d", lw=2, ms=5, label="inference")
    ax2.set_ylabel("inference ($\\mu$s / sample)", color="#c4622d")
    ax2.tick_params(axis="y", labelcolor="#c4622d")

    grid = sub[0]["grid"]
    note = "  (grid caps num_modes at 16)" if case.startswith("Burgers") else ""
    ax.set_title(f"{case}  (grid {grid}$^2$, 30 epochs){note}", fontsize=10)
    h1, a1 = ax.get_legend_handles_labels(); h2, a2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, a1 + a2, loc="upper center", fontsize=8, framealpha=.9)

fig.suptitle("Accuracy and inference cost against num_modes, precision highest, v5e", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=160)
print(f"wrote {OUT}")
