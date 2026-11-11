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

Each file carries the measurements themselves as well as the picture.  A second tab holds every
cell as a table: pin one of the three axes to a value and the other two become the rows and the
columns, so a slice can be read off as numbers rather than estimated from a marker.  The slice
on screen can be copied or saved as TSV.
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


def records(cells, why):
    """Every cell as one flat row, for the table the page builds client-side."""
    out = []
    for (batch, grid, modes), got in sorted(cells.items()):
        statuses = why.get((batch, grid, modes), {})
        row = {"b": batch, "n": grid, "m": modes}
        for engine in ENGINES + ["native"]:
            row[engine] = got.get(engine)
        if statuses:
            row["status"] = reason(statuses)
        out.append(row)
    return out


def ticks(values):
    import math
    vals = sorted(set(values))
    return {"tickmode": "array",
            "tickvals": [math.log2(v) if v > 1 else 0.0 for v in vals],
            "ticktext": [f"{v:,}" for v in vals]}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="__LIBRARY__" charset="utf-8"></script>
<style>
  :root { color-scheme: light; }
  body { margin: 0; background: #fff; color: #1a1a1a;
         font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  #plot { width: 100%; height: 820px; }
  .tabs { display: flex; gap: .4rem; justify-content: center; padding: .9rem 1rem .2rem;
          border-bottom: 1px solid #e3e3e3; }
  .tabs button { font: inherit; border: 1px solid #d8d8d8; background: #f7f7f8; color: #1a1a1a;
                 border-radius: 7px; padding: .35rem .95rem; cursor: pointer; }
  .tabs button:hover { border-color: #2f6fb2; }
  .tabs button.on { background: #2f6fb2; border-color: #2f6fb2; color: #fff; }
  .panel[hidden] { display: none; }
  .controls { display: flex; flex-wrap: wrap; gap: .8rem; align-items: flex-end;
              max-width: 70rem; margin: 1.2rem auto .4rem; padding: 0 1rem; }
  .controls label { display: flex; flex-direction: column; gap: .2rem; font-size: 12px;
                    color: #555; }
  .controls select, .controls button { font: inherit; padding: .3rem .5rem;
      border: 1px solid #d8d8d8; border-radius: 6px; background: #fff; color: #1a1a1a; }
  .controls button { cursor: pointer; background: #f7f7f8; }
  .controls button:hover { border-color: #2f6fb2; }
  .tablewrap { max-width: 100%; overflow-x: auto; padding: .6rem 1rem 2rem; }
  table.data { border-collapse: collapse; font-variant-numeric: tabular-nums;
               font-size: 12.5px; margin: 0 auto; }
  table.data th, table.data td { border: 1px solid #e3e3e3; padding: .28rem .55rem;
                                 text-align: right; white-space: nowrap; }
  table.data thead th { background: #f4f5f7; position: sticky; top: 0; }
  table.data th.corner { background: #eceef1; text-align: left; }
  table.data tbody th { background: #f4f5f7; text-align: right; }
  table.data td.na { color: #b4b4b4; }
  table.data td.win { font-weight: 700; }
  td.e-reim { color: #2f6fb2; } td.e-coupled { color: #c4622d; }
  td.e-reduced { color: #3f8f52; }
  .caption { max-width: 70rem; margin: 0 auto; padding: 0 1rem; color: #555; font-size: 12.5px; }
  .note { max-width: 54rem; margin: 0 auto 2rem; padding: 0 1rem; color: #444; }
  .note h2 { font-size: 1rem; margin: 1.4rem 0 .4rem; }
  .note dt { font-weight: 600; margin-top: .55rem; }
  .note dd { margin: 0 0 0 1.1rem; }
  code { background: #f3f3f3; padding: .1em .35em; border-radius: 3px; }
</style>
</head>
<body>
<div class="tabs">
  <button id="tabVol" class="on">Volume</button>
  <button id="tabDat">Data</button>
</div>

<div class="panel" id="panelVol">
  <div id="plot"></div>
  <div class="note">
  <h2>Reading this</h2>
  <p>Drag to rotate, scroll to zoom, double-click to reset. Click a legend entry to hide that
  engine. Every point is one measured cell; hover for the raw numbers. The Data tab has the
  same cells as a table.</p>
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
</div>

<div class="panel" id="panelDat" hidden>
  <div class="controls">
    <label>hold axis
      <select id="fix">
        <option value="n">grid N</option>
        <option value="m">num_modes</option>
        <option value="b">batch B</option>
      </select></label>
    <label>at value <select id="val"></select></label>
    <label>show <select id="metric">
        <option value="margin">margin, coloured by winner</option>
        <option value="best">fastest time, coloured by winner</option>
        <option value="winner">name of the fastest engine</option>
        <option value="reim">reim, us</option>
        <option value="coupled">coupled, us</option>
        <option value="reduced">reduced, us</option>
        <option value="native">native, us</option>
      </select></label>
    <button id="copy">Copy TSV</button>
    <button id="save">Save TSV</button>
  </div>
  <p class="caption" id="caption"></p>
  <div class="tablewrap" id="tablewrap"></div>
</div>

<script>
const DATA = __DATA__, LAYOUT = __LAYOUT__, REC = __RECORDS__;
const SLUG = "__SLUG__";
const AXIS = {b: "batch B", n: "grid N", m: "num_modes"};
const ENG = ["reim", "coupled", "reduced"];
let drawn = false;

function showVol() {
  document.getElementById("panelVol").hidden = false;
  document.getElementById("panelDat").hidden = true;
  document.getElementById("tabVol").classList.add("on");
  document.getElementById("tabDat").classList.remove("on");
  if (!drawn) {
    Plotly.newPlot("plot", DATA, LAYOUT,
      {responsive: true, displaylogo: false,
       toImageButtonOptions: {format: "png", scale: 2}});
    drawn = true;
  } else { Plotly.Plots.resize(document.getElementById("plot")); }
}
function showDat() {
  document.getElementById("panelVol").hidden = true;
  document.getElementById("panelDat").hidden = false;
  document.getElementById("tabDat").classList.add("on");
  document.getElementById("tabVol").classList.remove("on");
  build();
}
const uniq = k => [...new Set(REC.map(r => r[k]))].sort((a, b) => a - b);
const num = v => v >= 10000 ? v.toLocaleString(undefined, {maximumFractionDigits: 0})
                            : v.toLocaleString(undefined, {minimumFractionDigits: 1,
                                                           maximumFractionDigits: 1});

function fillValues() {
  const fix = document.getElementById("fix").value, sel = document.getElementById("val");
  const keep = sel.value;
  sel.innerHTML = "";
  for (const v of uniq(fix)) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v.toLocaleString(); sel.add(o);
  }
  if ([...sel.options].some(o => o.value === keep)) sel.value = keep;
}

// One cell: its text, a css class, and the plain value for the TSV copy.
function cell(r, metric) {
  if (!r) return {t: "—", cls: "na", raw: ""};
  const timed = ENG.filter(e => r[e] != null);
  if (!timed.length) return {t: "—", cls: "na", raw: "", tip: r.status || ""};
  const order = timed.slice().sort((a, b) => r[a] - r[b]);
  const best = order[0];
  if (metric === "winner") return {t: best, cls: "e-" + best, raw: best};
  if (metric === "best") return {t: num(r[best]), cls: "e-" + best, raw: r[best]};
  if (metric === "margin") {
    if (order.length < 2) return {t: "—", cls: "na", raw: ""};
    const g = r[order[1]] / r[order[0]];
    return {t: g.toFixed(2) + "×", cls: "win e-" + best, raw: g.toFixed(4),
            tip: best + " first at " + num(r[best]) + " us"};
  }
  const v = r[metric];
  if (v == null) return {t: "—", cls: "na", raw: "", tip: r.status || "not measured"};
  return {t: num(v), cls: metric === best ? "win e-" + best : "", raw: v};
}

let lastTsv = "";

function build() {
  const fix = document.getElementById("fix").value;
  const val = +document.getElementById("val").value;
  const metric = document.getElementById("metric").value;
  const [rk, ck] = ["b", "n", "m"].filter(k => k !== fix);
  const rows = uniq(rk), cols = uniq(ck);
  const at = new Map();
  for (const r of REC) if (r[fix] === val) at.set(r[rk] + "|" + r[ck], r);

  const head = ['<table class="data"><thead><tr>',
    `<th class="corner">${AXIS[rk]} \\ ${AXIS[ck]}</th>`];
  for (const c of cols) head.push(`<th>${c.toLocaleString()}</th>`);
  head.push("</tr></thead><tbody>");

  const tsv = [[`${AXIS[rk]}\\${AXIS[ck]}`, ...cols].join("\t")];
  let present = 0;
  for (const rv of rows) {
    head.push(`<tr><th>${rv.toLocaleString()}</th>`);
    const line = [rv];
    for (const cv of cols) {
      const c = cell(at.get(rv + "|" + cv), metric);
      if (c.raw !== "") present++;
      const tip = c.tip ? ` title="${c.tip}"` : "";
      head.push(`<td class="${c.cls}"${tip}>${c.t}</td>`);
      line.push(c.raw);
    }
    head.push("</tr>");
    tsv.push(line.join("\t"));
  }
  head.push("</tbody></table>");
  document.getElementById("tablewrap").innerHTML = head.join("");
  lastTsv = tsv.join("\n");

  const names = {winner: "the name of the fastest engine",
    best: "the fastest of the three times in microseconds, coloured by which engine it was",
    margin: "how far the winner beat the runner-up, coloured by which engine won",
    reim: "reim's time in microseconds, bold where it won",
    coupled: "coupled's time in microseconds, bold where it won",
    reduced: "reduced's time in microseconds, bold where it won",
    native: "the library transform's time in microseconds"};
  const total = rows.length * cols.length;
  let text = `${AXIS[fix]} held at ${val.toLocaleString()}. Rows are ${AXIS[rk]}, columns `
    + `are ${AXIS[ck]}, and each entry is ${names[metric]}.`;
  if (present === total) {
    text += ` All ${total} positions are measured.`;
  } else {
    text += ` ${present} of ${total} positions are measured. The gaps are not failures: each `
      + `grid was swept over its own log-spaced band ladder, so N = 512 was measured at 106 `
      + `and 232 while N = 256 was measured at 64 and 128, and a table whose columns are the `
      + `union of all nine ladders can only be sparse. Hold grid N or num_modes instead for a `
      + `table with no gaps.`;
  }
  document.getElementById("caption").textContent = text;
}

document.getElementById("tabVol").onclick = showVol;
document.getElementById("tabDat").onclick = showDat;
document.getElementById("fix").onchange = () => { fillValues(); build(); };
document.getElementById("val").onchange = build;
document.getElementById("metric").onchange = build;
document.getElementById("copy").onclick = async (e) => {
  try { await navigator.clipboard.writeText(lastTsv); e.target.textContent = "Copied"; }
  catch (err) { e.target.textContent = "Copy blocked"; }
  setTimeout(() => { e.target.textContent = "Copy TSV"; }, 1400);
};
document.getElementById("save").onclick = () => {
  const fix = document.getElementById("fix").value;
  const val = document.getElementById("val").value;
  const metric = document.getElementById("metric").value;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lastTsv], {type: "text/tab-separated-values"}));
  a.download = `${SLUG}_${fix}${val}_${metric}.tsv`;
  a.click(); URL.revokeObjectURL(a.href);
};
fillValues(); showVol();
</script>
</body>
</html>
"""


def page(title, subtitle, data, cells, rows, slug):
    """The standalone file: the volume on one tab, every measured cell on the other."""
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
    return (PAGE.replace("__TITLE__", title)
                .replace("__LIBRARY__", LIBRARY)
                .replace("__SLUG__", slug)
                .replace("__DATA__", json.dumps(data))
                .replace("__LAYOUT__", json.dumps(layout))
                .replace("__RECORDS__", json.dumps(rows)))


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
                         sub, data, cells, records(cells, why),
                         name[:-len(".html")]))
                written.append(out)
                print(f"wrote {out}  ({sum(len(t['x']) for t in data)} points)", flush=True)
    if not written:
        raise SystemExit("no batch sweeps found; run the batch benches first")
    print(f"\n{len(written)} viewers; open any of them in a browser", flush=True)


if __name__ == "__main__":
    main()
