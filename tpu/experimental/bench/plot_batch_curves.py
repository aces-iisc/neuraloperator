"""Device time per sample against batch, from the sweep docs_final records.

The curve is the argument for B*: per-sample cost falls while the free axis is starved and rises
again once the batch stops fitting, so the minimum is a real choice rather than "bigger is
better".  Stars mark it.
"""
import collections
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/train300/docs_final.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tpu/results/train300/batch_curves.png"
COLOURS = ["#2f6fb2", "#c4622d", "#3f8f52", "#8a5bb5"]


def main():
    rows = [r for r in csv.DictReader(open(CSV)) if r.get("config") == "sweep"]
    # the sweep is recorded twice per case, f32 then bf16, in file order
    series = collections.OrderedDict()
    for r in rows:
        case, batch = r["case"], int(r["batch"])
        seen = series.setdefault(case, [collections.OrderedDict(), collections.OrderedDict()])
        half = 1 if batch in seen[0] else 0
        seen[half][batch] = float(r["us_per_sample"])

    fig, axes = plt.subplots(1, len(series), figsize=(4.3 * len(series), 3.9), sharex=False)
    axes = [axes] if len(series) == 1 else list(axes)
    for ax, (case, (f32, bf16)) in zip(axes, series.items()):
        for (name, data), colour in zip((("float32", f32), ("bfloat16", bf16)), COLOURS):
            if not data:
                continue
            xs = sorted(data)
            ys = [data[x] for x in xs]
            ax.plot(xs, ys, "o-", color=colour, lw=2, ms=5, label=name)
            best = min(range(len(ys)), key=lambda i: ys[i])
            ax.plot(xs[best], ys[best], "*", color=colour, ms=16, zorder=5)
            ax.annotate(f"$B^*$={xs[best]}", (xs[best], ys[best]), textcoords="offset points",
                        xytext=(6, 8), fontsize=8, color=colour)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("batch")
        ax.set_ylabel("device time ($\\mu$s / sample)")
        ax.set_title(case, fontsize=10)
        ax.grid(alpha=.25, which="both")
        ax.legend(fontsize=8, framealpha=.9)

    fig.suptitle("Device time per sample against batch, trained weights, full test split",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
