"""Emit the LaTeX bodies of the GPU tables, so every figure has its numbers written down.

A figure shows a shape and a table shows a value, and the two answer different questions: the
sweep panels make it obvious that the batch matters and hopeless to read off what to set it to.
Each table here is the data behind one figure, generated from the same CSV the figure plots so
the two cannot drift apart.

Printed to stdout rather than written into the document: the paper is edited by hand and a
generator that rewrites it would fight that. Run it, read it, paste it.
"""
import collections
import csv
import os
import sys

BASE = os.environ.get("GPU_RESULTS", "tpu/results/gpu5070")
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
SHORT = {"float32": "\\texttt{f32}", "bfloat16": "\\texttt{bf16}"}
TRAINED = {"Burgers": 16, "Darcy-32": 16, "Darcy-64": 16, "NS-128": 64}


def rows(name):
    path = os.path.join(BASE, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    return list(csv.DictReader(open(path)))


def block(title, caption, label, spec, header, body, note=""):
    print(f"\n%% ---- {title} " + "-" * max(0, 60 - len(title)))
    print("\\begin{table}[h]\n\\centering")
    print(f"\\caption{{{caption}}}")
    print(f"\\label{{{label}}}\n\\small")
    print(f"\\begin{{tabular}}{{{spec}}}\n\\toprule")
    print(header)
    print("\\midrule")
    for line in body:
        print(line)
    print("\\bottomrule\n\\end{tabular}")
    if note:
        print(note)
    print("\\end{table}")


def batch_table():
    """What the formula says the batch should be, and what the machine says it is.

    Every batch in the table is priced against the same thing -- the cheapest per-sample time
    seen by any means in that session -- so the columns are comparable down the row. The
    33-point grid gets a price too, because it is a measurement like the others and not an
    oracle: the tuner refines between ladder rungs and lands on batches the grid has no point
    at, so the grid can be the one that is beaten.
    """
    data = {(r["case"], r["dtype"]): r for r in rows("tune_live_gpu.csv")}
    body = []
    for case in ORDER:
        for dtype in ("float32", "bfloat16"):
            r = data.get((case, dtype))
            if not r:
                continue
            floor = float(r["us_sweep"])
            form = f"${float(r['us_form']) / floor:.3f}\\times$" if r["us_form"] else "---"
            tuned = float(r["us_best"]) / floor
            grid = float(r["us_grid"]) / floor
            name = case if dtype == "float32" else ""
            body.append(
                f"{name:10} & {SHORT[dtype]} & {r['capacity']:>5} & "
                f"{r['b_form']:>5} & {form:>14} & "
                f"{r['b_best']:>5} & ${tuned:.3f}\\times$ & "
                f"{r['b_grid']:>5} & ${grid:.3f}\\times$ \\\\")
        body.append("\\addlinespace")
    body.pop()
    block("batch: predicted against measured",
          "The predicted batch against the measured one, on an RTX 5070. $B^{*}$ is the closed "
          "form on its own; ``tuner'' is what the whole search returns, ladder then refinement; "
          "``grid'' is the best of a 33-point sweep run in the same session on the same model. "
          "Each is priced against the cheapest per-sample time anything in that session reached, "
          "so $1.000\\times$ means nothing beat it. The grid is priced too, and loses: "
          "refinement lands between its points.",
          "tab:gpubatch", "@{}l l r rr rr rr@{}",
          "Dataset & & capacity & $B^{*}$ & costs & tuner & costs & grid & costs \\\\", body)


def search_cost_table():
    """A tuner is only worth running if it is cheaper than the sweep it replaces."""
    data = rows("tune_live_gpu.csv")
    body = []
    for case in ORDER:
        for dtype in ("float32", "bfloat16"):
            r = next((x for x in data if x["case"] == case and x["dtype"] == dtype), None)
            if not r:
                continue
            probes = int(r["probes_ladder"]) + int(r["probes_refine"])
            name = case if dtype == "float32" else ""
            body.append(f"{name:10} & {SHORT[dtype]} & {r['probes_ladder']:>2} & "
                        f"{r['probes_refine']:>2} & {probes:>3} & {r['search_s']:>6} & "
                        f"{r['sweep_points']:>3} & {r['sweep_s']:>6} & "
                        f"${float(r['sweep_s']) / float(r['search_s']):.1f}\\times$ \\\\")
    totals = (sum(int(r["probes_ladder"]) + int(r["probes_refine"]) for r in data),
              sum(float(r["search_s"]) for r in data),
              sum(int(r["sweep_points"]) for r in data),
              sum(float(r["sweep_s"]) for r in data))
    body.append("\\midrule")
    body.append(f"\\textbf{{total}} & & & & \\textbf{{{totals[0]}}} & "
                f"\\textbf{{{totals[1]:.1f}}} & \\textbf{{{totals[2]}}} & "
                f"\\textbf{{{totals[3]:.1f}}} & $\\mathbf{{{totals[3] / totals[1]:.1f}\\times}}$ \\\\")
    block("search cost",
          "What the search costs against what it replaces. ``Ladder'' and ``refine'' are the "
          "configurations the tuner timed; ``sweep'' is the 33-point exhaustive sweep it is "
          "scored against. Both were timed in the same session on the same model.",
          "tab:gpusearch", "@{}l l rrr r rr r@{}",
          "Dataset & & ladder & refine & timed & seconds & swept & seconds & saving \\\\", body)


def sweep_table():
    """The data behind the two batch-sweep panels: both implementations on one ladder."""
    data = {(r["case"], r["impl"]): r for r in rows("sweep_pair_summary.csv")}
    body = []
    for case in ORDER:
        for impl, shown in (("ours torch", "ours"), ("native torch", "native")):
            r = data.get((case, impl))
            if not r:
                continue
            name = case if impl == "ours torch" else ""
            body.append(f"{name:10} & {shown:7} & {r['capacity']:>5} & {r['best_batch']:>5} & "
                        f"{float(r['best_us']):8.2f} & {r['tuner_pick']:>5} & "
                        f"${float(r['tuner_loss']):.3f}\\times$ \\\\")
        body.append("\\addlinespace")
    body.pop()
    block("batch sweep, both spellings",
          "The data behind Figures~\\ref{fig:gpusweep} and~\\ref{fig:gpusweepnorm}: both torch "
          "spellings swept over one batch ladder in one session, \\texttt{float32}, on the "
          "trained weights. The capacity bound is ours and is applied unchanged to both, which "
          "is what makes the last column a test of it.",
          "tab:gpusweepdata", "@{}l l rrr rr@{}",
          "Dataset & & capacity & best $B$ & $\\mu$s/sample & ladder & ladder costs \\\\", body)


def modes_table():
    """The data behind the band panels: what each band is worth and what it costs."""
    acc = collections.defaultdict(dict)
    for r in rows("mode_accuracy_gpu.csv"):
        acc[r["case"]][int(r["n_modes"])] = r
    cost = collections.defaultdict(dict)
    for r in rows("mode_sweep_gpu.csv"):
        cost[r["case"]][int(r["n_modes"])] = float(r["ours_us_per_sample"])
    body = []
    for case in ORDER:
        curve = acc.get(case)
        if not curve:
            continue
        best = min(curve, key=lambda m: float(curve[m]["rel_h1"]))
        body.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{case}}}}} \\\\")
        for m in sorted(curve):
            r = curve[m]
            us = cost.get(case, {}).get(m)
            h1 = f"{float(r['rel_h1']):.4f}"
            cell = f"\\textbf{{{h1}}}" if m == best else h1
            body.append(
                f"\\quad {m:>3} & {int(r['params']):>11,} & {cell:>15} & "
                f"{float(r['rel_l2']):.4f} & " + (f"{us:8.2f}" if us else "---") + " \\\\")
        body.append("\\addlinespace")
    body.pop()
    block("band: accuracy and cost",
          "The data behind the first two panels of Figure~\\ref{fig:gpumodebatch}. Every row is "
          "a separate thirty-epoch training run under one protocol, so the only thing that "
          "differs down a block is \\nmodes; the error is the best test error over the run. "
          "Device time is from the cost sweep, at each value's own tuned batch. Bold is the "
          "lowest $H^1$ in the block, which for Burgers and Darcy-64 is inside the "
          "seed-to-seed spread of several of its neighbours.",
          "tab:gpumodedata", "@{}l r rr r@{}",
          "\nmodes & parameters & $H^1$ & rel.\\ $L^2$ & $\\mu$s/sample \\\\", body)


def engines_table():
    """The data behind the engine panels, at the band and width each dataset actually uses."""
    at = {}
    widths = set()
    for r in rows("engines_gpu.csv"):
        if r["status"] != "ok":
            continue
        at[(r["case"], int(r["n_modes"]), int(r["k_cols"]), r["engine"])] = float(r["us"])
        widths.add(int(r["k_cols"]))
    wide = max(widths)
    body = []
    for case in ORDER:
        m = TRAINED[case]
        for k in sorted(widths):
            base = at.get((case, m, k, "reim"))
            if base is None:
                continue
            cells = []
            for engine in ("coupled", "reduced"):
                v = at.get((case, m, k, engine))
                cells.append(f"{v:8.2f} & ${v / base:.2f}\\times$" if v
                             else "\\multicolumn{2}{c}{---}")
            name = f"{case} (\\nmodes$={m}$)" if k == min(widths) else ""
            body.append(f"{name:20} & {k:>5} & {base:8.2f} & " + " & ".join(cells) + " \\\\")
        body.append("\\addlinespace")
    body.pop()
    block("engines",
          f"The data behind Figure~\\ref{{fig:gpuengines}}, at each dataset's trained \nmodes and "
          f"every width measured, up to {wide} signals. Times are microseconds for one row "
          f"transform, best of fifty with CUDA events. A dash is a configuration the scheme "
          f"cannot run: the even/odd split needs an even row count and Burgers has 17.",
          "tab:gpuenginedata", "@{}l r r rr rr@{}",
          "Dataset & signals & reim & coupled & vs reim & reduced & vs reim \\\\", body)


def summary_table():
    """The whole tuner in one table: the band it picks, then the batch it picks at that band.

    The batch columns come from the end-to-end run at the chosen band, so what is reported is
    what the search returns rather than what one stage of it proposes.
    """
    import statistics
    acc = collections.defaultdict(dict)
    for r in rows("mode_accuracy_gpu.csv"):
        acc[r["case"]][int(r["n_modes"])] = float(r["rel_h1"])
    groups = collections.defaultdict(list)
    for r in rows("mode_noise_gpu.csv"):
        groups[(r["case"], r["n_modes"])].append(float(r["rel_h1"]))
    per_case = collections.defaultdict(list)
    for (case, _), seen in groups.items():
        if len(seen) > 1:
            per_case[case].append(max(seen) / min(seen))
    spread = {c: max(v) for c, v in per_case.items()}
    fallback = statistics.median(spread.values())
    tuned = {r["case"]: r for r in rows("tune_live_knee_gpu.csv") if r["dtype"] == "float32"}
    # What the energy criterion returns; it reads the targets, so it is the same on any machine.
    criterion = {"Burgers": 16, "Darcy-32": 24, "Darcy-64": 24, "NS-128": 24}

    body = []
    for case in ORDER:
        curve = acc[case]
        tol = spread.get(case, fallback)
        best = min(curve, key=curve.get)
        knee = min(b for b in curve if curve[b] <= tol * curve[best])
        t = tuned[case]
        body.append(
            f"{case:10} & {criterion[case]:>3} & {best:>3} & {knee:>3} & "
            f"$\\pm {100 * (tol - 1):.0f}\\%$ & "
            f"{t['b_best']:>4} & {t['b_grid']:>4} & "
            f"${float(t['us_best']) / float(t['us_sweep']):.3f}\\times$ \\\\")
    block("the tuner's choices",
          "The tuner's choices on an RTX 5070 against an exhaustive search of each, "
          "\\texttt{float32}. \nmodes\\ is scored twice: against the value that trained to the "
          "lowest test $H^1$, and against the smallest value whose error that "
          "dataset's own seed spread cannot separate from that best. ``Seeds'' is how far the "
          "dataset moves between four training runs differing only in seed; Navier--Stokes was "
          "too expensive to replicate, so it takes the median measured spread. The batch is "
          "tuned at the smallest value, and ``cost'' prices the tuner's batch against the cheapest "
          "per-sample time anything in that session reached.",
          "tab:gputuner", "@{}l rrrc rrr@{}",
          "& \\multicolumn{4}{c}{\nmodes} & \\multicolumn{3}{c}{batch $B$ at that \nmodes} \\\\\n"
          "\\cmidrule(lr){2-5}\\cmidrule(lr){6-8}\n"
          "Dataset & tuner & best & smallest & seeds & tuner & grid & cost \\\\", body)


TABLES = {"summary": summary_table, "batch": batch_table, "search": search_cost_table, "sweep": sweep_table,
          "modes": modes_table, "engines": engines_table}


def main():
    wanted = sys.argv[1:] or list(TABLES)
    for name in wanted:
        if name not in TABLES:
            raise SystemExit(f"unknown table {name}; have {', '.join(TABLES)}")
        TABLES[name]()


if __name__ == "__main__":
    main()
