"""The numbers behind the engine phase diagram, as text.

Three questions, in the order they decide anything:

  1. In the band -- the regime a layer runs in -- does any packing scheme ever win, and where?
  2. At the full spectrum, where does the staged plan hand a packing scheme the lead, and does
     the plan's leaf predict it?
  3. Is the restricted matrix worth building at all, against the vendor library?
"""
import collections
import csv
import os
import sys

GPU = ("RTX 5070", "tpu/results/gpu5070/engines_phase_gpu.csv", "graph_us", 16)
TPU = ("TPU v5e", "tpu/results/enginephase/engines_phase_tpu.csv", "device_us", 128)
ENGINES = ["reim", "coupled", "reduced"]
K_COLS = os.environ.get("PHASE_PLOT_K", "256")


def leaf(length, radix):
    """The direct DFT the staged plan finishes on, after peeling radix-sized stages."""
    stages = 0
    while length > radix and length % radix == 0:
        length //= radix
        stages += 1
    return stages, length


def load(path, column):
    if not os.path.exists(path):
        return None
    cells = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if row["k_cols"] == K_COLS and row.get(column):
            cells[(row["family"], int(row["grid"]), int(row["n_modes"]))][row["engine"]] = \
                float(row[column])
    return cells


def report(name, cells, radix):
    print(f"\n{'=' * 78}\n{name}   (radix {radix}, K = {K_COLS})\n{'=' * 78}")

    band = {(g, m): c for (fam, g, m), c in cells.items() if fam == "band"}
    staged = {g: c for (fam, g, m), c in cells.items() if fam == "staged"}

    print("\n-- band: where a packing scheme beats reim ----------------------------------")
    beat = []
    for (grid, modes), c in sorted(band.items()):
        got = {e: c[e] for e in ENGINES if e in c}
        best = min(got, key=got.get)
        if best != "reim":
            beat.append((grid, modes, best, got["reim"] / got[best]))
    if beat:
        for grid, modes, who, ratio in beat:
            print(f"   N={grid:<6} num_modes={modes:<6} {who:<8} {ratio:.3f}x faster than reim")
    else:
        print("   never: reim is first in every cell")
    print(f"   ({len(beat)} of {len(band)} cells)")

    print("\n-- staged, full non-redundant spectrum ---------------------------------------")
    print(f"   {'N':>6} {'winner':>9} {'margin':>8}   {'reim plan':>16} {'reduced plan':>16}"
          f"   leaf rule")
    for grid in sorted(staged):
        got = {e: staged[grid][e] for e in ENGINES if e in staged[grid]}
        order = sorted(got, key=got.get)
        margin = got[order[1]] / got[order[0]]
        s_r, leaf_r = leaf(grid, radix)
        s_d, leaf_d = leaf(grid // 2, radix)
        rule = "reduced: starved leaf repaired" if leaf_d > leaf_r else "-"
        print(f"   {grid:>6} {order[0]:>9} {margin:>7.2f}x   "
              f"{f'{s_r} st, leaf {leaf_r}':>16} {f'{s_d} st, leaf {leaf_d}':>16}   {rule}")

    print("\n-- against the vendor library ------------------------------------------------")
    print(f"   {'N':>6}  largest num_modes where the restricted matrix still wins")
    for grid in sorted({g for g, _ in band}):
        ours_win = [m for (g, m), c in band.items() if g == grid and "native" in c
                    and min(c[e] for e in ENGINES if e in c) < c["native"]]
        print(f"   {grid:>6}  " + (f"{max(ours_win)}  (of {grid})" if ours_win else "none"))


def main():
    for name, path, column, radix in (GPU, TPU):
        cells = load(path, column)
        if cells:
            report(name, cells, radix)
        else:
            print(f"\n{name}: no sweep at {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
