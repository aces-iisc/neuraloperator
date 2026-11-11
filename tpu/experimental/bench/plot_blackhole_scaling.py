"""How far the transform scales across the Tensix on a Blackhole card.

  python -m tpu.experimental.bench.plot_blackhole_scaling [out.png]

The split is over the batch, and the matrix unit cannot take less than a 16-wide column tile,
so a shape offers exactly ceil(B / 16) pieces of work.  Past that the extra tiles have nothing
to do, which is the flat part of every curve.  Speedup is the single-tile instruction count over
the busiest tile's, so it measures the partition and not the clock -- the simulator has none.

The dashed diagonal is one tile one unit of work.  Curves sit on it until their own ceiling,
and the only departure from it is at the right-hand end, where 256 pieces over 120 tiles leaves
some tiles holding three and others two.
"""
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = "tenstorrent/fno/results/scaling_blackhole.csv"
USABLE_TENSIX = 120


def load(path):
    curves = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if not row["status"].startswith("ok"):
            continue
        key = (int(row["grid"]), int(row["B"]), int(row["col_tiles"]))
        curves[key][int(row["tiles_offered"])] = float(row["speedup"])
    return curves


def main(argv):
    out_path = argv[0] if argv else "tpu/results/blackhole_scaling.png"
    curves = load(SOURCE)
    if not curves:
        raise SystemExit(f"no usable rows in {SOURCE}")

    figure, panel = plt.subplots(figsize=(7.2, 5.2))
    offered = sorted({t for curve in curves.values() for t in curve})
    panel.plot(offered, offered, linestyle="--", linewidth=0.9, color="#888888", zorder=1)

    batches = sorted({key[1] for key in curves})
    shades = {batch: plt.cm.viridis(i / max(1, len(batches) - 1))
              for i, batch in enumerate(batches)}
    marks = {64: "o", 128: "s", 256: "^"}

    for (grid, batch, col_tiles), curve in sorted(curves.items()):
        xs = sorted(curve)
        panel.plot(xs, [curve[x] for x in xs], marker=marks.get(grid, "o"), markersize=4.5,
                   linewidth=1.3, color=shades[batch], zorder=3,
                   label=f"N={grid}, B={batch}")

    for batch in batches:
        ceiling = min((batch + 15) // 16, USABLE_TENSIX)
        panel.axvline(ceiling, color=shades[batch], linewidth=0.7, alpha=0.35, zorder=0)

    panel.set_xscale("log", base=2)
    panel.set_yscale("log", base=2)
    panel.set_xlabel("Tensix tiles offered")
    panel.set_ylabel("speedup over one tile  (MVMULs on one tile / on the busiest)")
    panel.grid(alpha=0.25, which="both", linewidth=0.4)
    panel.legend(fontsize=7.5, frameon=False, ncol=3, loc="upper left")
    figure.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
