"""Interactive 3-D views of which real-input transform wins over a (B, N, num_modes) volume.

The flat maps answered the question at one batch.  With the batch a third axis the answer is a
volume, and a volume cannot be read off a printed page -- so this writes standalone HTML that
opens in a browser and can be rotated, zoomed and sliced by hand.  No server and no network: the
plotting library sits beside the files, so they draw from disk and keep drawing.

  python -m tpu.experimental.bench.view_engine_volume [outdir]

Six files, the same six comparisons the flat figures make:

  engine_volume_gpu_1d.html         engine_volume_tpu_1d.html
  engine_volume_gpu_2d.html         engine_volume_tpu_2d.html
  engine_volume_gpu_1d_nolib.html
  engine_volume_gpu_2d_nolib.html

Encoding, in each file:

  colour and shape   which of the three engines was fastest in that cell
  marker size        how far it beat the runner-up, so a win by a percent is not drawn like a
                     win by a factor of four
  opacity            whether it also beat the library transform; a transparent marker means the
                     library computed the whole spectrum and still came first.  The `nolib`
                     files drop this question and draw every marker solid.
  grey marker        a cell that could not be measured, with the reason in the hover text:
                     the field set did not fit in memory, or the engine does not accept that
                     batch.  Drawn rather than omitted, so a hole is visible as a hole.

Axes are log2 of each quantity because all three are swept in powers, and every point carries
a hover label with the raw numbers and the times of all four transforms.
"""
import collections
import csv
import json
import os
import sys

ENGINES = ["reim", "coupled", "reduced"]
COLOUR = {"reim": "#2f6fb2", "coupled": "#c4622d", "reduced": "#3f8f52"}
SYMBOL = {"reim": "circle", "coupled": "square", "reduced": "diamond"}
# Kept beside the viewers rather than fetched, so a file opened from disk with no
# network still draws.  view_engine_volume does not download it; see the README.
LIBRARY = "plotly.min.js"

# Several candidate paths per device: run_all names a bench's output after the module, while a
# bench run by hand writes wherever its own env var said.  First one present wins.
SOURCES = [
    ("gpu", "RTX 5070", ["tpu/results/gpu5070/engines_batch_gpu.csv"], "graph_us"),
    ("tpu", "TPU v5e", ["tpu/results/repro/docs_enginebatch.csv",
                        "tpu/results/enginebatch/engines_batch_tpu.csv"], "device_us"),
]


def load(paths, column, dims):
    """{(B, N, m): {engine: us or None}} plus the status of each cell, for one dimensionality."""
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        return None, None
    cells = collections.defaultdict(dict)
    why = {}
    for row in csv.DictReader(open(path)):
        if int(row["dims"]) != dims or row.get("family") != "band":
            continue
        key = (int(row["B"]), int(row["grid"]), int(row["n_modes"]))
        value = row.get(column)
        cells[key][row["engine"]] = float(value) if value else None
        if row["status"] != "ok":
            why.setdefault(key, {})[row["engine"]] = row["status"]
    return (cells, why) if cells else (None, None)


def reason(statuses):
    """One phrase for why a cell has no winner, from the statuses its engines reported."""
    seen = set(statuses.values())
    if "skip:mem" in seen or "OOM" in seen:
        return "did not fit in memory"
    if any(s.startswith(("ValueError", "TypeError")) for s in seen):
        return "batch not accepted by every engine"
    return "; ".join(sorted(seen))[:60]


def size_of(margin):
    """Marker diameter from the margin over the runner-up: 1.0x small, 4x and beyond large."""
    import math
    return 5.0 + 17.0 * min(math.log2(max(margin, 1.0)), 2.0) / 2.0


def traces(cells, why, show_lib):
    """One scatter3d trace per engine, plus one for the cells that could not be measured."""
    points = {e: {"x": [], "y": [], "z": [], "s": [], "o": [], "t": []} for e in ENGINES}
    holes = {"x": [], "y": [], "z": [], "t": []}
    import math
    for (batch, grid, modes), got in sorted(cells.items()):
        timed = {e: v for e, v in got.items() if e in ENGINES and v}
        label = f"B = {batch:,}<br>N = {grid}<br>num_modes = {modes}"
        if len(timed) < 2:
            holes["x"].append(math.log2(batch) if batch > 1 else 0.0)
            holes["y"].append(math.log2(grid))
            holes["z"].append(math.log2(modes))
            holes["t"].append(f"{label}<br>no winner: {reason(why.get((batch, grid, modes), {}))}")
            continue
        order = sorted(timed, key=timed.get)
        winner, best = order[0], timed[order[0]]
        margin = timed[order[1]] / best
        native = got.get("native")
        beaten = show_lib and native and native < best
        rows = "<br>".join(
            f"{e}: {got[e]:,.1f} us" if got.get(e) else f"{e}: ---"
            for e in ENGINES + ["native"])
        p = points[winner]
        p["x"].append(math.log2(batch) if batch > 1 else 0.0)
        p["y"].append(math.log2(grid))
        p["z"].append(math.log2(modes))
        p["s"].append(size_of(margin))
        p["o"].append(0.25 if beaten else 0.92)
        p["t"].append(f"{label}<br><b>{winner} wins by {margin:.2f}x</b>"
                      f"{'<br>but the library beat all three' if beaten else ''}<br>{rows}")

    out = []
    for engine in ENGINES:
        p = points[engine]
        if not p["x"]:
            continue
        # scatter3d carries one symbol and one opacity per trace, not per point, so the
        # library question needs two traces: solid for the cells where ours was first
        # outright, open for the cells where it led the three but the library led it.
        groups = [([i for i, o in enumerate(p["o"]) if o >= 0.5], SYMBOL[engine], engine, 0.92),
                  ([i for i, o in enumerate(p["o"]) if o < 0.5],
                   f"{SYMBOL[engine]}-open", f"{engine}, library faster", 1.0)]
        for picked, symbol, name, alpha in groups:
            if not picked:
                continue
            out.append({
                "type": "scatter3d", "mode": "markers", "name": name,
                "x": [p["x"][i] for i in picked], "y": [p["y"][i] for i in picked],
                "z": [p["z"][i] for i in picked],
                "text": [p["t"][i] for i in picked], "hoverinfo": "text",
                "marker": {"size": [p["s"][i] for i in picked], "color": COLOUR[engine],
                           "symbol": symbol, "opacity": alpha,
                           "line": {"width": 2.0 if "open" in symbol else 0.5,
                                    "color": "#ffffff"}},
            })
    if holes["x"]:
        out.append({
            "type": "scatter3d", "mode": "markers", "name": "not measurable",
            "x": holes["x"], "y": holes["y"], "z": holes["z"],
            "text": holes["t"], "hoverinfo": "text",
            "marker": {"size": 3.0, "color": "#b0b0b0", "symbol": "x", "opacity": 0.5},
        })
    return out


def ticks(values):
    import math
    vals = sorted(set(values))
    return {"tickmode": "array",
            "tickvals": [math.log2(v) if v > 1 else 0.0 for v in vals],
            "ticktext": [f"{v:,}" for v in vals]}


def page(title, subtitle, data, cells):
    batches = {b for b, _, _ in cells}
    grids = {n for _, n, _ in cells}
    modes = {m for _, _, m in cells}
    layout = {
        "title": {"text": f"{title}<br><sub>{subtitle}</sub>", "x": 0.5,
                  "font": {"size": 17}},
        "scene": {
            "xaxis": dict({"title": {"text": "batch B"}}, **ticks(batches)),
            "yaxis": dict({"title": {"text": "grid N"}}, **ticks(grids)),
            "zaxis": dict({"title": {"text": "num_modes"}}, **ticks(modes)),
            "camera": {"eye": {"x": 1.75, "y": 1.75, "z": 1.0}},
            "aspectmode": "cube",
        },
        "legend": {"orientation": "h", "y": -0.04, "x": 0.5, "xanchor": "center"},
        "margin": {"l": 0, "r": 0, "t": 76, "b": 0},
        "paper_bgcolor": "#ffffff",
        "height": 820,
    }
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="{LIBRARY}" charset="utf-8"></script>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #fff; color: #1a1a1a;
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
  #plot {{ width: 100%; height: 820px; }}
  .note {{ max-width: 54rem; margin: 0 auto 2rem; padding: 0 1rem; color: #444; }}
  .note h2 {{ font-size: 1rem; margin: 1.4rem 0 .4rem; }}
  .note dt {{ font-weight: 600; margin-top: .55rem; }}
  .note dd {{ margin: 0 0 0 1.1rem; }}
  code {{ background: #f3f3f3; padding: .1em .35em; border-radius: 3px; }}
</style>
</head>
<body>
<div id="plot"></div>
<div class="note">
<h2>Reading this</h2>
<p>Drag to rotate, scroll to zoom, double-click to reset. Click a legend entry to hide that
engine. Every point is one measured cell; hover for the raw numbers.</p>
<dl>
  <dt>Colour and shape</dt><dd>which of the three engines was fastest in that cell.</dd>
  <dt>Size</dt><dd>how far it beat the runner-up &mdash; smallest is a tie, largest is
  four times or more.</dd>
  <dt>Hollow marker</dt><dd>the library transform beat all three, so computing only the band
  was not worth it there. The shape still names which of ours came first.</dd>
  <dt>Small grey cross</dt><dd>a cell with no winner. Hover says why: the field set did not fit
  in memory, or an engine does not accept that batch.</dd>
</dl>
<p>All three axes are logarithmic, since all three are swept in powers.</p>
</div>
<script>
  Plotly.newPlot("plot", {json.dumps(data)}, {json.dumps(layout)},
                 {{responsive: true, displaylogo: false,
                   toImageButtonOptions: {{format: "png", scale: 2}}}});
</script>
</body>
</html>
"""


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "tpu/results/volumes"
    os.makedirs(outdir, exist_ok=True)
    written = []
    for tag, label, paths, column in SOURCES:
        for dims in (1, 2):
            cells, why = load(paths, column, dims)
            if not cells:
                print(f"  no {dims}-D data under {' or '.join(paths)}", flush=True)
                continue
            variants = [(True, "")] if tag == "tpu" else [(True, ""), (False, "_nolib")]
            for show_lib, suffix in variants:
                sub = ("marker size is the margin over the runner-up; hollow means "
                       "the library beat all three"
                       if show_lib else
                       "the three engines against each other only, the library left out")
                name = f"engine_volume_{tag}_{dims}d{suffix}.html"
                out = os.path.join(outdir, name)
                data = traces(cells, why, show_lib)
                open(out, "w").write(
                    page(f"{label} &mdash; {dims}-D transform over (B, N, num_modes)",
                         sub, data, cells))
                written.append(out)
                print(f"wrote {out}  ({sum(len(t['x']) for t in data)} points)", flush=True)
    if not written:
        raise SystemExit("no batch sweeps found; run the batch benches first")
    print(f"\n{len(written)} viewers; open any of them in a browser", flush=True)


if __name__ == "__main__":
    main()
