"""The numbers behind the dense engine grid, in 1-D and 2-D, as text.

Three questions, in the order they decide anything:

  1. Does any packing scheme ever beat reim in the band, and where does the boundary fall?
  2. At the full spectrum, where does the staged plan hand a packing scheme the lead, and does
     the plan's leaf predict it?
  3. Is the restricted matrix worth building at all, against the vendor library?
"""
import collections
import csv
import os

DEVICES = [("RTX 5070", "tpu/results/gpu5070/engines_grid_gpu.csv", "graph_us", 16),
           ("TPU v5e", "tpu/results/enginephase/engines_grid_tpu.csv", "device_us", 128)]
ENGINES = ["reim", "coupled", "reduced"]


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
        if row.get(column):
            key = (int(row["dims"]), row["family"], int(row["grid"]), int(row["n_modes"]))
            cells[key][row["engine"]] = float(row[column])
    return cells


def band_report(cells, dims):
    band = {(g, m): c for (d, fam, g, m), c in cells.items()
            if d == dims and fam == "band"}
    if not band:
        return
    tally, exceptions, library = collections.Counter(), collections.defaultdict(list), {}
    for (grid, modes), c in sorted(band.items()):
        got = {e: c[e] for e in ENGINES if e in c}
        best = min(got, key=got.get)
        tally[best] += 1
        if best != "reim":
            exceptions[grid].append((modes, best, got["reim"] / got[best]))
        if "native" in c and got[best] < c["native"]:
            library[grid] = max(library.get(grid, 0), modes)

    print(f"\n-- {dims}-D band: {len(band)} cells   "
          + "  ".join(f"{k} {v}" for k, v in tally.most_common()))
    if exceptions:
        for grid in sorted(exceptions):
            got = exceptions[grid]
            top = max(got, key=lambda t: t[2])
            print(f"   N={grid:<6} reim loses from num_modes={min(m for m, _, _ in got)} "
                  f"({len(got)} cells); best {top[1]} {top[2]:.3f}x at num_modes={top[0]}")
    else:
        print("   reim is first in every cell")
    print("   largest num_modes where the restricted matrix still beats the library:")
    print("     " + "  ".join(f"{g}:{library.get(g, '-')}"
                              for g in sorted({g for g, _ in band})))


def staged_report(cells, radix):
    staged = {g: c for (d, fam, g, m), c in cells.items() if fam == "staged"}
    if not staged:
        return
    print("\n-- 1-D staged, full non-redundant spectrum -----------------------------------")
    print(f"   {'N':>6} {'winner':>9} {'margin':>8}   {'reim plan':>16} {'reduced plan':>16}"
          f"   leaf rule")
    for grid in sorted(staged):
        got = {e: staged[grid][e] for e in ENGINES if e in staged[grid]}
        order = sorted(got, key=got.get)
        s_r, leaf_r = leaf(grid, radix)
        s_d, leaf_d = leaf(grid // 2, radix)
        rule = "reduced: starved leaf repaired" if leaf_d > leaf_r else "-"
        print(f"   {grid:>6} {order[0]:>9} {got[order[1]] / got[order[0]]:>7.2f}x   "
              f"{f'{s_r} st, leaf {leaf_r}':>16} {f'{s_d} st, leaf {leaf_d}':>16}   {rule}")


def main():
    for name, path, column, radix in DEVICES:
        cells = load(path, column)
        if not cells:
            print(f"\n{name}: no sweep at {path}")
            continue
        print(f"\n{'=' * 78}\n{name}   (radix {radix})\n{'=' * 78}")
        for dims in (1, 2):
            band_report(cells, dims)
        staged_report(cells, radix)


if __name__ == "__main__":
    main()
