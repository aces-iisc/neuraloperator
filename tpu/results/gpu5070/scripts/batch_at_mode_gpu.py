"""The second half of the two-step choice: pick the band, then sweep the batch at that band.

The batch sweeps already in the repository all run at the band the shipped weights were trained
at. That is the right band to time a released model at and the wrong one to answer with here,
because the band is itself a choice and the two choices interact: a wider band grows the working
set, which moves the batch where the set stops fitting. Sweeping the batch at the band the
accuracy study actually selects is what makes the pair of figures a recipe rather than two
separate measurements.

The band selected is the knee, not the argmin. Test error against band falls steeply and then
flattens, so the argmin sits somewhere out on the flat where it is decided by training noise,
and taking it buys an unmeasurable amount of accuracy for a band that costs real time. The knee
-- the narrowest band whose error is within the noise of the best -- is the band a person would
train with, so it is the band the batch is swept at. Both are recorded and the figure marks
both, so the choice is visible rather than asserted.

The noise is the measured one. A companion script trains the same band under several seeds and
the spread it returns becomes the tolerance here, so the threshold is a property of the
experiment rather than a round number chosen after seeing the curve.

Weights are random here. The batch's cost does not depend on what the weights contain, only on
their shape, and training a model per band twice over is time spent to reach the same number.
"""
import collections
import csv
import os
import statistics
import sys

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune
from torch_fno.fno.model import FNO

DEV = torch.device("cuda")
ACCEL = detect()
ACC_CSV = os.environ.get("MODE_ACC_OUT", "/home/anirudh/fno/mode_accuracy_gpu.csv")
OUT = os.environ.get("BATCH_AT_MODE_OUT", "/home/anirudh/fno/batch_at_mode_gpu.csv")
SPLIT = 1000
REPS = 30
WARMUP = 5
NOISE_CSV = os.environ.get("MODE_NOISE_OUT", "/home/anirudh/fno/mode_noise_gpu.csv")
FALLBACK_TOLERANCE = 1.01
SHAPE = {"Burgers": ((17, 16), 24), "Darcy-32": ((32, 32), 24),
         "Darcy-64": ((64, 64), 24), "NS-128": ((128, 128), 64)}
FIELDS = ["target", "case", "n_modes", "role", "channels", "batch", "us_per_sample",
          "capacity", "is_candidate"]

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def tolerance_from_noise(path):
    """Each dataset's own seed-to-seed spread. Inside it, two bands are not distinguishable.

    Per dataset rather than one number for all of them, because the spreads are not alike:
    Burgers reruns itself to within 1.28x and the Darcy sets are far tighter. Taking the widest
    and applying it everywhere would hand the tight datasets a tolerance they never earned and
    pick a band several steps too narrow for them.

    A dataset with no seed study -- Navier--Stokes, where one run is a third of an hour -- takes
    the median of the measured spreads, not the widest. The widest belongs to Burgers, which has
    800 training samples against 5000 for the Darcy sets and 10000 for Navier--Stokes, and a
    small set is exactly why its reruns scatter. Handing that scatter to the largest dataset in
    the study would be borrowing from the one least like it.
    """
    if not os.path.exists(path):
        print(f"no seed study at {path}; falling back to {FALLBACK_TOLERANCE}", flush=True)
        return collections.defaultdict(lambda: FALLBACK_TOLERANCE)
    groups = collections.defaultdict(list)
    for row in csv.DictReader(open(path)):
        groups[(row["case"], row["n_modes"])].append(float(row["rel_h1"]))
    per_case = collections.defaultdict(list)
    for (case, _), errors in groups.items():
        if len(errors) > 1:
            per_case[case].append(max(errors) / min(errors))
    spread = {case: max(v) for case, v in per_case.items()}
    fallback = statistics.median(spread.values()) if spread else FALLBACK_TOLERANCE
    for case, value in sorted(spread.items()):
        print(f"  seed spread {case:9} {value:.4f}x", flush=True)
    print(f"  unmeasured datasets take the median, {fallback:.4f}x", flush=True)
    return collections.defaultdict(lambda: fallback, spread)


def bands_from_accuracy(path, tolerance):
    """{case: (knee, argmin, {band: h1})} -- the band to sweep and the one it was chosen over."""
    by_case = collections.defaultdict(dict)
    for row in csv.DictReader(open(path)):
        by_case[row["case"]][int(row["n_modes"])] = float(row["rel_h1"])
    out = {}
    for case, errors in by_case.items():
        best = min(errors, key=errors.get)
        knee = min(b for b in errors if errors[b] <= tolerance[case] * errors[best])
        out[case] = (knee, best, errors)
    return out


def grid_for(cap):
    """Geometric plus a fine tail: enough to locate the knee without assuming a grain."""
    out = {1, 2, 4, 8}
    step = max(1, cap // 16)
    out |= {k * step for k in range(1, 25)}
    out |= {int(cap * f) for f in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)}
    return sorted(b for b in out if 1 <= b <= SPLIT)


def timed(fn, x, reps=REPS):
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(x)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        best = float("inf")
        for _ in range(reps):
            begin.record()
            fn(x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def sweep(case, modes, rows):
    (height, width), channels = SHAPE[case]
    model = FNO(1, 1, channels, modes, 4, "dense").to(DEV).eval()
    model.prebuild()
    tuner = AutoTune(ModelSpec(grid=max(height, width), channels=channels, n_modes=modes),
                     ACCEL, max_batch=SPLIT)
    capacity = tuner.capacity("float32")
    candidates = {p.batch for p in tuner.candidates("float32")}
    print(f"\n{case}: m={modes}  capacity={capacity}  candidates={sorted(candidates)}",
          flush=True)

    for batch in sorted(set(grid_for(capacity)) | {b for b in candidates if b <= SPLIT}):
        x = torch.randn(batch, 1, height, width, device=DEV)
        try:
            us = timed(model, x) / batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  B={batch:<5} out of memory", flush=True)
            continue
        rows.append({"target": ACCEL.name, "case": case, "n_modes": modes, "role": "sweep",
                     "channels": channels, "batch": batch, "us_per_sample": f"{us:.4f}",
                     "capacity": capacity, "is_candidate": int(batch in candidates)})
        print(f"  B={batch:<5} {us:9.3f} us/sample{'  *' if batch in candidates else ''}",
              flush=True)
        del x
    del model
    torch.cuda.empty_cache()


def main():
    chosen = bands_from_accuracy(ACC_CSV, tolerance_from_noise(NOISE_CSV))
    rows = []
    for case in ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]:
        if case not in chosen:
            print(f"{case}: no accuracy rows yet, skipping", flush=True)
            continue
        knee, best, errors = chosen[case]
        print(f"\n{'=' * 70}\n{case}: knee m={knee} (H1 {errors[knee]:.4f}), "
              f"argmin m={best} (H1 {errors[best]:.4f})", flush=True)
        sweep(case, knee, rows)

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}  ({len(rows)} rows)", flush=True)

    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        best = min(sub, key=lambda r: float(r["us_per_sample"]))
        cands = [r for r in sub if int(r["is_candidate"])]
        pick = min(cands, key=lambda r: float(r["us_per_sample"])) if cands else None
        line = (f"{case:9} m={sub[0]['n_modes']:>3}  best B={best['batch']:>4} "
                f"({float(best['us_per_sample']):.3f} us/sa)")
        if pick:
            line += (f"   ladder picks B={pick['batch']:>4}, costing "
                     f"{float(pick['us_per_sample']) / float(best['us_per_sample']):.3f}x")
        print(line, flush=True)


if __name__ == "__main__":
    main()
