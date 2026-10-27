"""Run the tuner for real: call `best()` with a live timer and see what comes back.

Every tuner score in this repository so far was computed by asking `candidates()` for its ladder
and looking each rung up in a sweep that had already been run. That is not what the tuner does.
`best()` times the ladder and then bisects between its rungs, and the gate it refines under is

    if not refine or (grain and self.capacity(dtype) >= grain): return ranked[0]

so refinement is skipped only where a batch grain has been measured AND capacity reaches it.
`batch_grain` is `None` on every GPU in `arch.py`, deliberately, because no batch quantum has
been measured on one. On a GPU the first clause is therefore false and **the shipped tuner always
refines**. Scoring the ladder alone measures a component the tuner never returns on its own.

So this script hands `best()` a real CUDA-event timer and records what it actually probes. One
pass gives both numbers -- the ladder's own winner is the cheapest rung among the batches the
tuner proposed, and the refined winner is whatever `best()` returns after bisecting -- so the two
are compared on identical measurements rather than across two sessions.

It also records the one quantity a tuner is useless without: how long it took. A search that
costs more than the sweep it replaces has not saved anything.

The ground truth is swept in the same process, immediately after, on the same model object and
through the same timer. Scoring against a sweep from an earlier session does not work here: the
first attempt did, and returned ratios below one -- the tuner apparently beating the optimum --
because the machine was simply in a different state that day. The effect was worst in bfloat16,
where the stored sweep was up to 13% slower at identical batches. A tuner is being asked whether
it found the best batch, and that question only means something against times it could have
measured itself.
"""
import collections
import csv
import os
import sys
import time

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune
from torch_fno.fno.model import FNO

DEV = torch.device("cuda")
ACCEL = detect()
OUT = os.environ.get("TUNE_LIVE_OUT", "/home/anirudh/fno/tune_live_gpu.csv")
# Which band to tune at. The shipped one by default, so these scores sit beside the other
# measurements of the released model; BANDS=knee instead reads the band the accuracy study
# selected, which is what the band-then-batch figure needs to mark the tuner's real answer.
BANDS = os.environ.get("TUNE_LIVE_BANDS", "shipped")
ACC_CSV = os.environ.get("MODE_ACC_OUT", "/home/anirudh/fno/mode_accuracy_gpu.csv")
NOISE_CSV = os.environ.get("MODE_NOISE_OUT", "/home/anirudh/fno/mode_noise_gpu.csv")
SPLIT = 1000
REPS = int(os.environ.get("TUNE_LIVE_REPS", 20))
WARMUP = 5
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
# The shipped configurations, so these scores sit beside the ladder-only ones already reported
# rather than beside a different model.
CASES = [("Burgers", (17, 16), 24, 16), ("Darcy-32", (32, 32), 24, 16),
         ("Darcy-64", (64, 64), 24, 16), ("NS-128", (128, 128), 64, 64)]
FIELDS = ["target", "case", "dtype", "capacity", "grain", "refined",
          "b_form", "b_ladder", "b_best", "b_sweep", "b_grid",
          "us_form", "us_ladder", "us_best", "us_sweep", "us_grid",
          "probes_ladder", "probes_refine", "sweep_points", "search_s", "sweep_s"]

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def timed(model, x, reps=REPS):
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        best = float("inf")
        for _ in range(reps):
            begin.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def dense_grid(capacity):
    """Geometric plus a fine tail, the same shape of ladder the other GPU sweeps use."""
    out = {1, 2, 4, 8}
    step = max(1, capacity // 16)
    out |= {k * step for k in range(1, 25)}
    out |= {int(capacity * f) for f in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)}
    return sorted(b for b in out if 1 <= b <= SPLIT)


def run(case, shape, channels, modes, dtype_name, rows):
    height, width = shape
    torch_dtype = DTYPES[dtype_name]
    model = FNO(1, 1, channels, modes, 4, "dense").to(DEV).eval()
    model.set_dtype(torch_dtype).prebuild(torch_dtype)
    tuner = AutoTune(ModelSpec(grid=max(shape), channels=channels, n_modes=modes),
                     ACCEL, max_batch=SPLIT)

    probed, order = {}, []

    def measure(plan):
        """What the tuner is allowed to know: the cost of one batch, nothing else."""
        x = torch.randn(plan.batch, 1, height, width, device=DEV)
        try:
            us = timed(model, x) / plan.batch
        finally:
            del x
        probed[plan.batch] = us
        order.append(plan.batch)
        return us

    ladder = {p.batch for p in tuner.candidates(dtype_name)}
    form = tuner.batch(dtype_name)
    form = form[0] if isinstance(form, tuple) else form
    capacity = tuner.capacity(dtype_name)

    start = time.perf_counter()
    us_best, plan = tuner.best(measure, dtype_name)
    elapsed = time.perf_counter() - start

    # The ladder's own answer, read off the same measurements the tuner just took.
    seen_ladder = {b: probed[b] for b in ladder if b in probed}
    b_ladder = min(seen_ladder, key=seen_ladder.get)
    n_ladder = len(seen_ladder)
    refined = len(probed) > n_ladder

    # The exhaustive answer, measured now, on this model, through this timer.
    sweep_start = time.perf_counter()
    swept = {}
    for batch in dense_grid(capacity):
        if batch in probed:
            swept[batch] = probed[batch]
            continue
        x = torch.randn(batch, 1, height, width, device=DEV)
        try:
            swept[batch] = timed(model, x) / batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue
        finally:
            del x
    sweep_s = time.perf_counter() - sweep_start
    # Ground truth is the best of everything measured this session, the grid and the tuner's own
    # probes together. Refinement lands between rungs and the grid has no point there, so a grid
    # alone can be beaten by the thing it is supposed to be judging -- which shows up as a cost
    # below one and means the yardstick was short, not that the tuner found something impossible.
    # Two yardsticks, because they answer different questions. The grid alone answers the one
    # that matters -- can an exhaustive sweep find a batch the tuner missed -- and the tuner's
    # probes are excluded from it so the comparison is not against itself. The union is the best
    # batch known by the end and is what the cost column is quoted against.
    b_grid = min(swept, key=swept.get)
    us_grid = swept[b_grid]
    swept.update(probed)
    b_sweep = min(swept, key=swept.get)

    rows.append({"target": ACCEL.name, "case": case, "dtype": dtype_name,
                 "capacity": capacity, "grain": ACCEL.batch_grain, "refined": int(refined),
                 "b_form": form, "b_ladder": b_ladder, "b_best": plan.batch,
                 "b_sweep": b_sweep, "b_grid": b_grid,
                 "us_form": f"{swept[form]:.4f}" if form in swept else "",
                 "us_ladder": f"{seen_ladder[b_ladder]:.4f}", "us_best": f"{us_best:.4f}",
                 "us_sweep": f"{swept[b_sweep]:.4f}", "us_grid": f"{us_grid:.4f}",
                 "probes_ladder": n_ladder, "probes_refine": len(probed) - n_ladder,
                 "sweep_points": len(swept), "search_s": f"{elapsed:.2f}",
                 "sweep_s": f"{sweep_s:.2f}"})
    floor = swept[b_sweep]
    print(f"  {dtype_name:9} capacity {capacity:5d}  ladder B={b_ladder:<5} "
          f"-> refined B={plan.batch:<5} vs swept B={b_sweep:<5}   "
          f"ladder {seen_ladder[b_ladder] / floor:.3f}x  refined {us_best / floor:.3f}x   "
          f"{n_ladder}+{len(probed) - n_ladder} probes in {elapsed:.1f}s against "
          f"{len(swept)} in {sweep_s:.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()


def knee_bands():
    """The band the accuracy study selected, by the same rule the batch sweep used."""
    import statistics
    errors = collections.defaultdict(dict)
    for row in csv.DictReader(open(ACC_CSV)):
        errors[row["case"]][int(row["n_modes"])] = float(row["rel_h1"])
    groups = collections.defaultdict(list)
    for row in csv.DictReader(open(NOISE_CSV)):
        groups[(row["case"], row["n_modes"])].append(float(row["rel_h1"]))
    per_case = collections.defaultdict(list)
    for (case, _), seen in groups.items():
        if len(seen) > 1:
            per_case[case].append(max(seen) / min(seen))
    spread = {case: max(v) for case, v in per_case.items()}
    fallback = statistics.median(spread.values()) if spread else 1.01
    out = {}
    for case, curve in errors.items():
        tol = spread.get(case, fallback)
        best = min(curve, key=curve.get)
        out[case] = min(b for b in curve if curve[b] <= tol * curve[best])
    return out


def main():
    print(f"target: {ACCEL.name}, batch_grain={ACCEL.batch_grain}, bands={BANDS}\n")
    cases = CASES
    if BANDS == "knee":
        chosen = knee_bands()
        cases = [(c, shape, ch, chosen.get(c, m)) for c, shape, ch, m in CASES]
    rows = []
    for case, shape, channels, modes in cases:
        print(f"{case} (grid {shape}, channels {channels}, modes {modes})", flush=True)
        for dtype_name in DTYPES:
            run(case, shape, channels, modes, dtype_name, rows)

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {OUT}", flush=True)

    print(f"\n{'case':9} {'dtype':9} {'form':>6} {'ladder':>7} {'refined':>8} {'swept':>6}  "
          f"{'form':>8} {'ladder':>8} {'refined':>8}   {'search':>8} {'sweep':>8}")
    print("-" * 100)
    worst = {"form": 1.0, "ladder": 1.0, "refined": 1.0}
    for row in rows:
        floor = float(row["us_sweep"])
        cf = float(row["us_form"]) / floor if row["us_form"] else None
        cl, cr = float(row["us_ladder"]) / floor, float(row["us_best"]) / floor
        if cf:
            worst["form"] = max(worst["form"], cf)
        worst["ladder"] = max(worst["ladder"], cl)
        worst["refined"] = max(worst["refined"], cr)
        print(f"{row['case']:9} {row['dtype']:9} {row['b_form']:6} {row['b_ladder']:7} "
              f"{row['b_best']:8} {row['b_sweep']:6}  "
              f"{(f'{cf:.3f}x' if cf else 'n/a'):>8} {cl:7.3f}x {cr:7.3f}x   "
              f"{row['search_s']:>7}s {row['sweep_s']:>7}s")
    print("-" * 100)
    print(f"worst against a same-session sweep: closed form {worst['form']:.3f}x, "
          f"ladder {worst['ladder']:.3f}x, refined {worst['refined']:.3f}x")
    beaten = [r for r in rows if float(r["us_grid"]) < float(r["us_best"])]
    gains = [float(r["us_grid"]) / float(r["us_best"]) for r in rows]
    print(f"the dense grid beat the tuner on {len(beaten)}/{len(rows)} configurations; "
          f"the tuner beat the grid by up to {max(gains):.3f}x")
    print(f"search {sum(float(r['search_s']) for r in rows):.1f}s against "
          f"{sum(float(r['sweep_s']) for r in rows):.1f}s of sweeping")


if __name__ == "__main__":
    main()
