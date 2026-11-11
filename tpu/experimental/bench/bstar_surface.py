"""Where the per-sample cost stops falling, over a dense (grid, num_modes) plane.

B* is the smallest batch whose per-sample cost is within a tolerance of the cheapest batch
measured.  The argmin alone is not usable: once the batch saturates the machine the tail is
flat to within run-to-run noise, so the argmin lands wherever the noise fell and overstates
how much batch the shape actually needs.

A curve whose knee sits past the widest batch in the sweep is reported as censored, with B*
as a lower bound.  That happens for the whole small-N end of the 1-D plane, where one sample
is a few hundred elements and no batch in reach saturates anything.  Dropping those cells
would misreport a sweep that did not go far enough as a shape that needs no batch.

Reused by plot_bstar_surface.py and by the autotuner work, so the definition lives here once.
"""
import collections
import csv
import os

# Per-sample element count of one spectral cell: what the batch has to multiply to fill the
# machine.  The row pass of a 2-D transform is the 1-D program run N times, so the budget
# carries the extra factor of the grid.
def working_set(dims, grid, n_modes):
    return grid * n_modes * (grid if dims == 2 else 1)


DEVICES = {
    "gpu": ("RTX 5070", ["tpu/results/gpu5070/engines_batch_gpu.csv"], "graph_us"),
    "tpu": ("TPU v5e", ["tpu/results/repro/docs_enginebatch.csv",
                        "tpu/results/enginebatch/engines_batch_tpu.csv"], "device_us"),
}
ENGINES = ["reim", "coupled", "reduced", "native"]
MIN_POINTS = 5
TOLERANCE = float(os.environ.get("BSTAR_TOL", "0.05"))


class Knee(collections.namedtuple("Knee", "b_star b_argmin us_per_sample censored ladder_top")):
    """Where one batch curve flattens.  censored means the knee is past the widest batch tried."""


def load_curves(device, dims):
    """{(family, grid, n_modes, engine): {batch: us_per_sample}} for one dimensionality."""
    _, paths, column = DEVICES[device]
    path = next((p for p in paths if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"no batch sweep for {device}: tried {paths}")
    curves = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        if int(row["dims"]) != dims or row.get("status") != "ok" or not row.get(column):
            continue
        total_us, batch = float(row[column]), int(row["B"])
        if total_us <= 0:
            continue
        key = (row["family"], int(row["grid"]), int(row["n_modes"]), row["engine"])
        curves[key][batch] = total_us / batch
    return curves, column


def knee_of(curve, tolerance=TOLERANCE):
    cheapest = min(curve.values())
    b_star = min(b for b, us in curve.items() if us <= cheapest * (1.0 + tolerance))
    ladder_top = max(curve)
    return Knee(b_star, min(curve, key=curve.get), cheapest, b_star >= ladder_top, ladder_top)


def surface(device, dims, tolerance=TOLERANCE):
    """{(family, grid, n_modes, engine): Knee} over every curve with enough points to have a knee."""
    curves, _ = load_curves(device, dims)
    return {key: knee_of(curve, tolerance)
            for key, curve in curves.items() if len(curve) >= MIN_POINTS}


FIELDS = ["device", "dims", "family", "grid", "n_modes", "engine", "working_set",
          "b_star", "b_argmin", "us_per_sample", "censored", "ladder_top"]


def write_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def table(device, dims, tolerance=TOLERANCE):
    rows = []
    for (family, grid, n_modes, engine), knee in sorted(surface(device, dims, tolerance).items()):
        rows.append({"device": device, "dims": dims, "family": family, "grid": grid,
                     "n_modes": n_modes, "engine": engine,
                     "working_set": working_set(dims, grid, n_modes),
                     "b_star": knee.b_star, "b_argmin": knee.b_argmin,
                     "us_per_sample": f"{knee.us_per_sample:.6g}",
                     "censored": int(knee.censored), "ladder_top": knee.ladder_top})
    return rows


def main():
    out_dir = "tpu/results/bstar_surface"
    os.makedirs(out_dir, exist_ok=True)
    for device in DEVICES:
        for dims in (1, 2):
            rows = table(device, dims)
            if not rows:
                continue
            path = os.path.join(out_dir, f"bstar_{device}_{dims}d.csv")
            write_csv(path, rows)
            censored = sum(r["censored"] for r in rows)
            print(f"{path}: {len(rows)} cells, {len(rows) - censored} resolved, {censored} censored")


if __name__ == "__main__":
    main()
