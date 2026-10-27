"""Dense batch sweep for both torch spellings on one ladder, in one session.

The earlier GPU sweep covered our port only, so each implementation in the comparison table
was timed at our optimum rather than its own.  This sweeps the same ladder for the textbook
spelling as well, which both fixes the comparison and tests whether the capacity model -- a
model of our dataflow -- transfers to a dataflow it was not written for.
"""
import csv
import sys

sys.path.insert(0, "/home/anirudh")

import torch

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune
from torch_fno.fno.model import FNO
from torch_fno.fno.layers import add_grid, gelu

DEV = torch.device("cuda")
ACCEL = detect()
WEIGHTS = "/home/anirudh/fno"
CASES = [("Burgers", (17, 16), 24, 16), ("Darcy-32", (32, 32), 24, 16),
         ("Darcy-64", (64, 64), 24, 16), ("NS-128", (128, 128), 64, 64)]
SPLIT = 1000
REPS = 30

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


def grid_for(cap):
    """Geometric plus a fine tail: enough to locate the knee without assuming a grain."""
    out = {1, 2, 4, 8}
    step = max(1, cap // 16)
    out |= {k * step for k in range(1, 25)}
    out |= {int(cap * f) for f in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)}
    return sorted(b for b in out if 1 <= b <= SPLIT)


def timed(f, x, reps=REPS):
    with torch.no_grad():
        for _ in range(5):
            f(x)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        best = float("inf")
        for _ in range(reps):
            begin.record()
            f(x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def native_model(ours):
    """rfft2 -> slice the band -> complex einsum -> irfft2, sharing our weights."""
    def run(x):
        half = ours.n_modes // 2
        x = ours.lift(add_grid(x))
        for blk in ours.blocks:
            n, c, h, w = x.shape
            weight = torch.complex(blk.spectral.w_re, blk.spectral.w_im)
            xf = torch.fft.rfft2(x, dim=(-2, -1))
            out = torch.zeros(n, weight.shape[3], h, w // 2 + 1, dtype=xf.dtype,
                              device=x.device)
            out[:, :, :half, :half] = torch.einsum("adio,niad->noad", weight[:half],
                                                   xf[:, :, :half, :half])
            out[:, :, h - half:, :half] = torch.einsum("adio,niad->noad", weight[half:],
                                                       xf[:, :, h - half:, :half])
            sp = torch.fft.irfft2(out, s=(h, w), dim=(-2, -1))
            x = gelu(sp + blk.skip(x))
        return ours.project(x)
    return run


rows, summary = [], []
for name, shape, ch, modes in CASES:
    spec = ModelSpec(grid=max(shape), channels=ch, n_modes=modes)
    tuner = AutoTune(spec, ACCEL, max_batch=SPLIT)
    cap = tuner.capacity("float32")
    cands = [p.batch for p in tuner.candidates("float32")]

    ours = FNO(1, 1, ch, modes, 4).to(DEV).eval()
    ours.load_state_dict(torch.load(f"{WEIGHTS}/{name}_state.pt", map_location="cpu"))
    ours = ours.to(DEV).eval()
    for b in ours.blocks:
        b.spectral.layout = "mode"
    ours.prebuild(torch.float32)

    print(f"\n{'=' * 72}\n{name}  grid {shape}  C={ch}  modes={modes}  "
          f"capacity {cap}\n{'=' * 72}", flush=True)

    for impl, f in (("ours torch", ours), ("native torch", native_model(ours))):
        series = {}
        for b in grid_for(cap):
            try:
                x = torch.randn(b, 1, *shape, device=DEV)
                series[b] = timed(f, x) / b
                del x
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
        if not series:
            continue
        best = min(series, key=series.get)
        seen = {b: series[b] for b in cands if b in series}
        pick = min(seen, key=seen.get) if seen else None
        loss = series[pick] / series[best] if pick else float("inf")
        print(f"  {impl:14} swept {len(series):3d} points, best B={best:4d} "
              f"({series[best]:8.3f} us/sa);  tuner picks {pick} -> {loss:.3f}x of best",
              flush=True)
        summary.append({"case": name, "impl": impl, "capacity": cap, "best_batch": best,
                        "best_us": series[best], "tuner_pick": pick,
                        "tuner_us": series[pick] if pick else None,
                        "tuner_loss": loss, "within_1pct": int(loss <= 1.01)})
        for b in sorted(series):
            rows.append({"target": ACCEL.name, "case": name, "impl": impl,
                         "dtype": "float32", "batch": b, "us_per_sample": series[b],
                         "capacity": cap, "is_candidate": int(b in cands),
                         "best": int(b == best)})
    del ours
    torch.cuda.empty_cache()

print(f"\n{'=' * 72}\nsummary: best batch and tuner score per implementation\n{'=' * 72}",
      flush=True)
print(f"{'case':10} {'impl':14} {'cap':>5} {'best B':>7} {'us/sa':>10} "
      f"{'pick':>6} {'loss':>7}", flush=True)
for s in summary:
    print(f"{s['case']:10} {s['impl']:14} {s['capacity']:5d} {s['best_batch']:7d} "
          f"{s['best_us']:10.3f} {str(s['tuner_pick']):>6} {s['tuner_loss']:6.3f}x",
          flush=True)
hits = sum(s["within_1pct"] for s in summary)
print(f"\ntuner within 1% of the measured optimum on {hits}/{len(summary)} "
      f"(implementation, case) pairs", flush=True)

for data, path in ((rows, "/home/anirudh/fno/sweep_pair.csv"),
                   (summary, "/home/anirudh/fno/sweep_pair_summary.csv")):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"wrote {path}", flush=True)
