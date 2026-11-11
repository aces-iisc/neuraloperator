"""The engine sweeps as LaTeX tables, so the figures have their numbers beside them.

The sweep holds thousands of cells per device per dimensionality, which is too many to print.
Each table therefore fixes the batch and reports one row per grid at two bands: a narrow one,
which is where an operator sits, and the full spectrum, which is where the packing schemes were
designed to be used. The library transform is carried alongside, since whether to restrict the
matrix at all is a separate question from which engine to restrict it with.

The batch is a constant, not a function of the grid. It differs between the two
dimensionalities because the dataset does: a 1-D cell is `N * B` elements, a 2-D cell is
`B * N^2`, so one 4096^2 field is 67 MiB and the widest batch that fits every grid in 2-D is
far narrower than in 1-D. Both are stated in the caption rather than derived.
"""
import collections
import csv
import os
import sys

def batches(device, one_d, two_d):
    """The constant batch this device was swept at, per dimensionality.

    Each device carries its own pair because each was swept on its own ladder.  The GPU sweep
    steps the batch in powers of two, so its columns sit at 2^14 and 2^4; the TPU record
    predates that and sits at the round numbers it was run with.
    """
    return {1: int(os.environ.get(f"{device}_B1", one_d)),
            2: int(os.environ.get(f"{device}_B2", two_d))}


SOURCES = {"tpu": ("TPU v5e", ["tpu/results/repro/docs_enginebatch.csv",
                               "tpu/results/enginebatch/engines_batch_tpu.csv"], "device_us",
                   batches("TPU", 20000, 20)),
           "gpu": ("RTX 5070", ["tpu/results/gpu5070/engines_batch_gpu.csv"], "graph_us",
                   batches("GPU", 16384, 16))}
ENGINES = ["reim", "coupled", "reduced", "native"]


def load(paths, column, dims, batch):
    """({(grid, n_modes): {engine: us}}, {(grid, n_modes, engine): status}) at one fixed batch.

    The second map carries the cells that produced no time.  A table that prints a bare dash
    leaves the reader to guess whether the cell failed or was never tried, so the reason
    travels with the data and the caption states it.
    """
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise SystemExit(f"no sweep found; looked for {paths}")
    cells = collections.defaultdict(dict)
    missing = {}
    for row in csv.DictReader(open(path)):
        if row["family"] != "band" or int(row["dims"]) != dims or int(row["B"]) != batch:
            continue
        key = (int(row["grid"]), int(row["n_modes"]))
        if row["status"] == "ok" and row.get(column):
            cells[key][row["engine"]] = float(row[column])
        else:
            missing[key + (row["engine"],)] = row["status"]
    return cells, missing


def absent_clause(missing, shown):
    """One sentence naming the cells the table leaves empty, or nothing if it leaves none."""
    hit = {k: v for k, v in missing.items() if k[:2] in shown}
    if not hit:
        return ""
    def group(keys):
        """Engines that share a cell are named together, so one cell reads as one phrase."""
        by_cell = collections.defaultdict(list)
        for g, m, engine in sorted(keys):
            by_cell[(g, m)].append(rf"\texttt{{{engine}}}")
        return ", ".join(f"{' and '.join(es)} at $N = {g}$, $\\nmodes = {m}$"
                         for (g, m), es in sorted(by_cell.items()))

    oom = [k for k, why in hit.items() if why == "OOM"]
    rest = [k for k, why in hit.items() if why != "OOM"]
    parts = []
    if oom:
        parts.append(f"A dash is a cell that ran out of device memory ({group(oom)}), so the "
                     "engine that fits where the others do not is itself a result.")
    if rest:
        reasons = ", ".join(sorted({hit[k] for k in rest}))
        parts.append(f"A dash is a cell the sweep could not time ({group(rest)}: {reasons}).")
    return " " + " ".join(parts)


def nearest(bands, target):
    return min(bands, key=lambda m: abs(m - target))


def fmt(value, winner):
    """Microseconds, bolded when this engine is first among the three."""
    text = f"{value:,.1f}" if value < 10000 else f"{value:,.0f}"
    return f"\\textbf{{{text}}}" if winner else text


def row_for(cells, grid, modes):
    got = cells[(grid, modes)]
    three = {e: got[e] for e in ENGINES[:3] if e in got}
    best = min(three, key=three.get) if three else None
    return " & ".join(fmt(got[e], e == best) if e in got else "---" for e in ENGINES)


def table(key, dims):
    label, paths, column, device_batch = SOURCES[key]
    batch = device_batch[dims]
    cells, missing = load(paths, column, dims, batch)
    grids = sorted({g for g, _ in cells})
    if not grids:
        raise SystemExit(f"{key} {dims}-D has no cells at B = {batch}")
    shown = set()
    for one in grids:
        bands = sorted(m for g, m in cells if g == one)
        shown.update({(one, nearest(bands, max(2, one // 8))), (one, bands[-1])})
    absent = absent_clause(missing, shown)
    unit = "signals" if dims == 1 else "fields"
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             rf"\caption{{Engine grid, {dims}-D, on {label}. Device time in microseconds for "
             r"one call at two bands: narrow ($\nmodes \approx N/8$) and full ($\nmodes = N$). "
             r"Bold is the fastest of the three engines; \texttt{native} is the library "
             rf"transform computing the whole spectrum. Every cell is {batch:,} {unit}, the "
             rf"same batch at every grid.{absent}}}",
             rf"\label{{tab:enginegrid{key}{dims}d}}",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}r rrrrr rrrrr@{}}", r"\toprule",
             r"& \multicolumn{5}{c}{\textbf{narrow band}} "
             r"& \multicolumn{5}{c}{\textbf{full band}} \\",
             r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}",
             r"$N$ & \nmodes & \texttt{reim} & \texttt{coupled} "
             r"& \texttt{reduced} & \texttt{native} "
             r"& \nmodes & \texttt{reim} & \texttt{coupled} & \texttt{reduced} "
             r"& \texttt{native} \\",
             r"\midrule"]
    for grid in grids:
        bands = sorted(m for g, m in cells if g == grid)
        narrow, full = nearest(bands, max(2, grid // 8)), bands[-1]
        lines.append(f"{grid} & {narrow} & {row_for(cells, grid, narrow)} "
                     f"& {full} & {row_for(cells, grid, full)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    for key in ("tpu", "gpu"):
        for dims in (1, 2):
            print(table(key, dims))
            print()


if __name__ == "__main__":
    main()
