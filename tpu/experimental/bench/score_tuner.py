"""Score the tuner against the dense sweeps: closed form, ladder coverage, and ladder pick.

Three quantities per configuration, all priced against the cheapest batch the sweep found: the
closed form on its own, the cheapest of the candidates the ladder offers, and whether the swept
optimum was on the ladder at all.  The row is written to a CSV so the scorecard figure plots the
same numbers the summary counts, and the GPU half of the scorecard is scored by the companion
script beside the GPU sweeps.
"""
import csv
import importlib.util
import sys
import types

pkg = types.ModuleType("tpu")
pkg.__path__ = ["tpu"]
sys.modules["tpu"] = pkg
spec = importlib.util.spec_from_file_location("tpu.arch", "tpu/arch.py")
arch = importlib.util.module_from_spec(spec)
sys.modules["tpu.arch"] = arch
spec.loader.exec_module(arch)
fno = types.ModuleType("tpu.fno")
fno.__path__ = ["tpu/fno"]
sys.modules["tpu.fno"] = fno
spec = importlib.util.spec_from_file_location("tpu.fno.autotune", "tpu/fno/autotune.py")
at = importlib.util.module_from_spec(spec)
sys.modules["tpu.fno.autotune"] = at
spec.loader.exec_module(at)

SHAPE = {"Burgers": (17, 24, 16), "Darcy-32": (32, 24, 16),
         "Darcy-64": (64, 24, 16), "NS-128": (128, 64, 64)}
ACCEL = {"dense_v5e": arch.TPU_V5E, "dense_v6e": arch.TPU_V6E}
OUT = "tpu/results/tuner_scorecard_tpu.csv"
FIELDS = ["target", "case", "dtype", "b_form", "b_pick", "b_best",
          "cost_form", "cost_pick", "form_swept", "offered"]

rows = []
print(f"{'gen':4} {'case':9} {'dtype':9} {'B form':>7} {'B pick':>7} {'B best':>7} "
      f"{'form':>8} {'pick':>8}  {'offered':>8}")
print("-" * 86)
n = exact_form = exact_pick = covered = 0
worst_form = worst_pick = 1.0
for gen, accel in ACCEL.items():
    series, grids = {}, {}
    for r in csv.DictReader(open(f"tpu/results/{gen}/tab_tuner.csv")):
        if r["role"] != "sweep" or r["status"] != "ok":
            continue
        series.setdefault((r["case"], r["dtype"]), {})[int(r["batch"])] = \
            float(r["us_per_sample"])
    for (case, dt), s in sorted(series.items()):
        grid, ch, modes = SHAPE[case]
        tuner = at.AutoTune(accel, arch.ModelSpec(grid=grid, channels=ch, n_modes=modes),
                            max_batch=max(s))
        form = tuner.batch(dt)
        form = form[0] if isinstance(form, tuple) else form
        cands = [p.batch for p in tuner.candidates(dt)]
        best = min(s, key=s.get)
        seen = {b: s[b] for b in cands if b in s}
        pick = min(seen, key=seen.get) if seen else None
        fl = s[form] / s[best] if form in s else None
        pl = s[pick] / s[best] if pick else None
        n += 1
        exact_form += (form == best)
        exact_pick += (pick == best)
        covered += (best in cands)
        if fl:
            worst_form = max(worst_form, fl)
        if pl:
            worst_pick = max(worst_pick, pl)
        rows.append({"target": f"TPU {gen[-3:]}", "case": case, "dtype": dt,
                     "b_form": form, "b_pick": pick, "b_best": best,
                     "cost_form": f"{fl:.5f}" if fl else "",
                     "cost_pick": f"{pl:.5f}" if pl else "",
                     "form_swept": int(form in s), "offered": int(best in cands)})
        print(f"{gen[-3:]:4} {case:9} {dt:9} {form:7d} {str(pick):>7} {best:7d} "
              f"{(f'{fl:.3f}x' if fl else 'n/a'):>8} {(f'{pl:.3f}x' if pl else 'n/a'):>8}"
              f"  {'yes' if best in cands else 'NO':>8}")
print("-" * 86)
print(f"closed form exact      {exact_form}/{n}   worst cost {worst_form:.3f}x")
print(f"ladder pick exact      {exact_pick}/{n}   worst cost {worst_pick:.3f}x")
print(f"optimum on the ladder  {covered}/{n}")

with open(OUT, "w", newline="") as fh:
    writer = csv.DictWriter(fh, FIELDS)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {OUT}")
