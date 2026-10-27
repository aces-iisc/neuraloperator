"""Does the predicted batch find the best time for our own code on the GPU?

Scored against two independent dense sweeps of the same implementation: the earlier one, which
covers both precisions, and the paired one, which covers float32 on a finer ladder. Three
quantities per configuration -- the closed form on its own, the candidate ladder's pick, and the
batch the sweep actually found cheapest.

Having two sweeps is what makes the scores readable rather than just low. The batch curve here
is a smooth bowl with no grain, so the floor is flat and the argmin is not a stable quantity:
the two sweeps disagree about the fastest batch on three of the four datasets. Pricing each
sweep's winner on the other measures that disagreement, and it comes out the same size as the
tuner's own penalty. The cost penalty is therefore the number that means something and the hit
rate is not, so both are written out and the scorecard figure plots the penalty.
"""
import collections
import csv
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune

ACCEL = detect()
SHAPE = {"Burgers": (17, 24, 16), "Darcy-32": (32, 24, 16),
         "Darcy-64": (64, 24, 16), "NS-128": (128, 64, 64)}
SWEEPS = [("gpu_sweep.csv", "earlier sweep, both precisions"),
          ("sweep_pair.csv", "paired sweep, float32")]
BASE = os.environ.get("GPU_RESULTS", "/home/anirudh/plot/tpu/results/gpu5070")
ORDER = ["Burgers", "Darcy-32", "Darcy-64", "NS-128"]
FIELDS = ["target", "case", "dtype", "sweep", "b_form", "b_pick", "b_best",
          "cost_form", "cost_pick", "form_swept", "offered"]


def load(path):
    """{(case, dtype): {batch: us_per_sample}} for our implementation only."""
    series = collections.defaultdict(dict)
    for row in csv.DictReader(open(f"{BASE}/{path}")):
        if row.get("impl", "ours torch") != "ours torch":
            continue
        series[(row["case"], row["dtype"])][int(row["batch"])] = float(row["us_per_sample"])
    return series


def score(path, title, rows):
    series = load(path)
    print(f"\n{'=' * 90}\n{title}  ({path})\n{'=' * 90}")
    print(f"{'case':9} {'dtype':9} {'cap':>5} {'B form':>7} {'B pick':>7} {'B best':>7} "
          f"{'form cost':>10} {'pick cost':>10}  {'offered':>8}")
    print("-" * 90)
    n = ex_form = ex_pick = covered = 0
    worst_form = worst_pick = 1.0
    for (case, dtype), s in sorted(series.items()):
        grid, ch, modes = SHAPE[case]
        tuner = AutoTune(ModelSpec(grid=grid, channels=ch, n_modes=modes), ACCEL,
                         max_batch=max(s))
        cap = tuner.capacity(dtype)
        form = tuner.batch(dtype)
        form = form[0] if isinstance(form, tuple) else form
        cands = [p.batch for p in tuner.candidates(dtype)]
        best = min(s, key=s.get)
        seen = {b: s[b] for b in cands if b in s}
        pick = min(seen, key=seen.get) if seen else None
        fc = s[form] / s[best] if form in s else None
        pc = s[pick] / s[best] if pick else None
        n += 1
        ex_form += (form == best)
        ex_pick += (pick == best)
        covered += (best in cands)
        if fc:
            worst_form = max(worst_form, fc)
        if pc:
            worst_pick = max(worst_pick, pc)
        rows.append({"target": ACCEL.name, "case": case, "dtype": dtype, "sweep": path,
                     "b_form": form, "b_pick": pick, "b_best": best,
                     "cost_form": f"{fc:.5f}" if fc else "",
                     "cost_pick": f"{pc:.5f}" if pc else "",
                     "form_swept": int(form in s), "offered": int(best in cands)})
        print(f"{case:9} {dtype:9} {cap:5d} {form:7d} {str(pick):>7} {best:7d} "
              f"{(f'{fc:.3f}x' if fc else 'not swept'):>10} "
              f"{(f'{pc:.3f}x' if pc else 'n/a'):>10}  "
              f"{'yes' if best in cands else 'NO':>8}")
    print("-" * 90)
    print(f"closed form exact        {ex_form}/{n}   worst {worst_form:.3f}x")
    print(f"ladder pick exact        {ex_pick}/{n}   worst {worst_pick:.3f}x")
    print(f"optimum on the ladder    {covered}/{n}")
    return n, ex_pick, worst_pick


def reproducibility():
    """Price each sweep's fastest batch on the other, for the float32 cases both cover.

    This is the yardstick the hit rate has to be read against: whatever this costs is what an
    exact argmin is worth on this hardware.
    """
    a, b = load(SWEEPS[0][0]), load(SWEEPS[1][0])
    out = []
    print(f"\n{'=' * 90}\nthe same code swept twice: do the sweeps agree?\n{'=' * 90}")
    print(f"{'case':9} {'argmin A':>9} {'argmin B':>9} {'B on A':>9} {'A on B':>9}")
    print("-" * 90)
    for case in ORDER:
        sa, sb = a.get((case, "float32")), b.get((case, "float32"))
        if not sa or not sb:
            continue
        ba, bb = min(sa, key=sa.get), min(sb, key=sb.get)
        # Each winner is priced on the other sweep's own floor, so both columns read as a
        # penalty over that sweep's best and are directly comparable to the tuner's.
        on_a = sa[bb] / sa[ba] if bb in sa else None
        on_b = sb[ba] / sb[bb] if ba in sb else None
        out.append({"case": case, "argmin_a": ba, "argmin_b": bb,
                    "cost_b_on_a": f"{on_a:.5f}" if on_a else "",
                    "cost_a_on_b": f"{on_b:.5f}" if on_b else ""})
        print(f"{case:9} {ba:9d} {bb:9d} "
              f"{(f'{on_a:.3f}x' if on_a else 'n/a'):>9} "
              f"{(f'{on_b:.3f}x' if on_b else 'n/a'):>9}")
    worst = max([float(r[k]) for r in out for k in ("cost_b_on_a", "cost_a_on_b") if r[k]],
                default=1.0)
    print("-" * 90)
    print(f"worst disagreement between the two sweeps  {worst:.3f}x")
    return out


def main():
    print(f"target: {ACCEL.name}")
    rows = []
    totals = [score(path, title, rows) for path, title in SWEEPS]
    n = sum(t[0] for t in totals)
    print(f"\nacross both sweeps: ladder pick exact {sum(t[1] for t in totals)}/{n}, "
          f"worst {max(t[2] for t in totals):.3f}x")
    repro = reproducibility()

    for data, fields, name in ((rows, FIELDS, "tuner_scorecard_gpu.csv"),
                               (repro, list(repro[0]), "tuner_reproducibility_gpu.csv")):
        with open(f"{BASE}/{name}", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fields)
            writer.writeheader()
            writer.writerows(data)
        print(f"wrote {BASE}/{name}")


if __name__ == "__main__":
    main()
