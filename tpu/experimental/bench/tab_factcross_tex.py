"""The three channel filters over each transform, as one LaTeX table.

Reads the cross written by docs_fact and keeps the two arms that share a filter and differ only
in the transform, so a row answers one question: what does replacing jnp.fft with our partial
transform cost or save, at a fixed filter.

  python -m tpu.experimental.bench.tab_factcross_tex <csv> [out.tex]
"""
import collections
import csv
import os
import sys

CASES = [("Burgers", "Burgers $17\\times16$"), ("Darcy-32", "Darcy $32^2$"),
         ("Darcy-64", "Darcy $64^2$"), ("NS-128", "NS $128^2$")]
FILTERS = [("dense", "dense"), ("ffno", "F-FNO"), ("tfno", "TFNO")]
# the two arms that share a filter and differ only in the transform
# ARMS names the two engine labels to compare, oldest runs included: the pair must share a
# filter and differ only in the transform, or the row stops isolating anything.
ARMS = [a.split(":") for a in
        os.environ.get("CROSS_ARMS", "baseline:jnp.fft,ours-xform:ours").split(",")]
TIME = os.environ.get("CROSS_TIME", "us_per_sample")

HEAD = r"""\begin{table}[H]
\centering
\small
\caption{The three channel filters over each transform, on one v5e. The filter is the same
object in both columns, so a row isolates the transform: \texttt{jnp.fft} is
\texttt{rfft2}/\texttt{irfft2} with the band truncated, ``ours'' the partial transform of
Section~\ref{sec:recursive}.
%(note)s Relative $L^2$ is on the test split
after %(steps)s steps of the objective each configuration names. $R = 8$.}
\label{tab:factcross}
\setlength{\tabcolsep}{7pt}
\begin{tabular}{@{}ll rr r rr@{}}
\toprule
& & \multicolumn{2}{c}{\textbf{$\mu$s/sample}} & &
\multicolumn{2}{c}{\textbf{rel $L^2$}} \\
\cmidrule(lr){3-4}\cmidrule(lr){6-7}
\textbf{Dataset} & \textbf{Filter} & \texttt{%(a)s} & %(b)s & \textbf{speedup}
& \texttt{jnp.fft} & ours \\
\midrule
"""
TAIL = r"""\bottomrule
\end{tabular}
\end{table}
"""


def load(path):
    rows = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if row["status"] != "ok":
            continue
        rows[(row["case"], row["variant"])][row["engine"]] = row
    return rows


def number(value, digits=2):
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "---"


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    table = load(path)
    body = []
    for case, case_label in CASES:
        got = [(f, lab) for f, lab in FILTERS if (case, f) in table]
        if not got:
            continue
        body.append(f"\\multirow{{{len(got)}}}{{*}}{{{case_label}}}\n")
        for filt, filt_label in got:
            cells = table[(case, filt)]
            times, errs = [], []
            for arm, _ in ARMS:
                row = cells.get(arm)
                times.append(float(row[TIME]) if row else None)
                errs.append(float(row["rel_l2"]) if row else None)
            gain = (f"${times[0] / times[1]:.2f}\\times$"
                    if times[0] and times[1] else "---")
            best = 0 if (times[0] or 1e9) <= (times[1] or 1e9) else 1
            cols = [f"\\textbf{{{number(t)}}}" if i == best else number(t)
                    for i, t in enumerate(times)]
            body.append(f" & {filt_label} & {cols[0]} & {cols[1]} & {gain} & "
                        f"{number(errs[0], 4)} & {number(errs[1], 4)} \\\\\n")
        body.append("\\midrule\n")
    if body and body[-1] == "\\midrule\n":
        body[-1] = ""
    steps = "60"
    text = (HEAD % {"steps": steps, "a": ARMS[0][1], "b": ARMS[1][1],
                "note": os.environ.get("CROSS_NOTE", "")}) + "".join(body) + TAIL
    if out:
        open(out, "w").write(text)
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
