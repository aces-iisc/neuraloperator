"""The engine-grid sweeps as LaTeX tables, so the figures have their numbers beside them.

The sweep holds 166 cells per device per dimensionality, which is too many to print. Each
table therefore reports one row per grid at two bands: a narrow one, which is where an
operator sits, and the full spectrum, which is where the packing schemes were designed to
be used. The library transform is carried alongside, since whether to restrict the matrix
at all is a separate question from which engine to restrict it with.
"""
import collections
import csv
import sys

SOURCES = {"tpu": ("TPU v5e", "tpu/results/enginephase/engines_grid_tpu.csv", "device_us"),
           "gpu": ("RTX 5070", "tpu/results/gpu5070/engines_grid_gpu.csv", "graph_us")}
ENGINES = ["reim", "coupled", "reduced", "native"]


def load(path, column, dims):
    """{(grid, n_modes): {engine: us}} plus {grid: width}, the batch axis for this dims."""
    cells, widths = collections.defaultdict(dict), {}
    key = "k_cols" if dims == 1 else "fields"
    for row in csv.DictReader(open(path)):
        if row["family"] != "band" or int(row["dims"]) != dims or row["status"] != "ok":
            continue
        grid, modes = int(row["grid"]), int(row["n_modes"])
        cells[(grid, modes)][row["engine"]] = float(row[column])
        if row.get(key):
            widths[grid] = int(row[key])
    return cells, widths


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
    label, path, column = SOURCES[key]
    cells, widths = load(path, column, dims)
    grids = sorted({g for g, _ in cells})
    axis = "$K$" if dims == 1 else "fields"
    budget = (r"Signal count $K$ holds the element count fixed at $NK = 2^{24}$, capped at "
              r"$2^{18}$." if dims == 1 else
              r"The field count holds the element count fixed on the same budget.")
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             rf"\caption{{Engine grid, {dims}-D, on {label}. Device time in microseconds for "
             r"one call at two bands: narrow ($\nmodes \approx N/8$) and full ($\nmodes = N$). "
             r"Bold is the fastest of the three engines; \texttt{native} is the library "
             rf"transform computing the whole spectrum. {budget}}}",
             rf"\label{{tab:enginegrid{key}{dims}d}}",
             r"\setlength{\tabcolsep}{3.5pt}",
             r"\begin{tabular}{@{}rr rrrrr rrrrr@{}}", r"\toprule",
             r"& & \multicolumn{5}{c}{\textbf{narrow band}} "
             r"& \multicolumn{5}{c}{\textbf{full band}} \\",
             r"\cmidrule(lr){3-7}\cmidrule(lr){8-12}",
             rf"$N$ & {axis} & \nmodes & \texttt{{reim}} & \texttt{{coupled}} "
             r"& \texttt{reduced} & \texttt{native} "
             r"& \nmodes & \texttt{reim} & \texttt{coupled} & \texttt{reduced} "
             r"& \texttt{native} \\",
             r"\midrule"]
    for grid in grids:
        bands = sorted(m for g, m in cells if g == grid)
        narrow, full = nearest(bands, max(2, grid // 8)), bands[-1]
        width = widths.get(grid)
        shown = f"{width:,}" if width else "---"
        lines.append(f"{grid} & {shown} & {narrow} & {row_for(cells, grid, narrow)} "
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
