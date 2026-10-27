"""Why is the reference implementation slower, measured rather than argued.

Three questions, three sections:

  A  which of its defaults costs the time        -- ablate one default at a time
  B  what the device actually runs               -- kernel buckets and launch counts
  C  is the shared batch unfair to it             -- sweep its own batch

Everything is timed by the same routine as the five-way table, so the only thing that
changes between rows is the model.
"""
import csv
import sys
import warnings

sys.path.insert(0, "/home/anirudh")
sys.path.insert(0, "/home/anirudh/fno/nop")
warnings.filterwarnings("ignore")

import torch
from torch.profiler import ProfilerActivity, profile

from neuralop.models import FNO as TheirFNO
from torch_fno.fno.model import FNO as OurFNO

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# (case, grid, channels, our n_modes, batch, their n_modes) -- same table as the five-way run.
CASES = [("Burgers", (17, 16), 24, 16, 396, 22), ("Darcy-32", (32, 32), 24, 16, 128, 22),
         ("Darcy-64", (64, 64), 24, 16, 32, 22), ("NS-128", (128, 128), 64, 64, 4, 78)]

# Each entry turns off exactly one of the reference implementation's defaults.
VARIANTS = [("default", {}),
            ("no channel mlp", {"use_channel_mlp": False}),
            ("mlp skip identity", {"channel_mlp_skip": "identity"}),
            ("no hermitian split", {"enforce_hermitian_symmetry": False}),
            ("no grid embedding", {"positional_embedding": None}),
            ("all four off", {"use_channel_mlp": False, "channel_mlp_skip": "identity",
                              "enforce_hermitian_symmetry": False,
                              "positional_embedding": None})]


def timed(f, x, reps=20, warm=8):
    with torch.no_grad():
        for _ in range(warm):
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


def bucket(name):
    low = name.lower()
    if any(k in low for k in ("gemm", "cutlass", "tensorop", "matmul", "sgemm", "bmm",
                              "ampere", "sm90", "sm100", "sm120", "wgmma", "mma")):
        return "tensor-core / GEMM"
    if any(k in low for k in ("fft", "spectral", "postprocess", "preprocess", "regular_fft",
                              "vector_fft")):
        return "cuFFT"
    if any(k in low for k in ("copy", "transpose", "permute", "contiguous", "reshape",
                              "roll", "shift")):
        return "copy / relayout"
    if "memcpy" in low or "memset" in low:
        return "DMA / memcpy"
    if any(k in low for k in ("elementwise", "vectorized", "unrolled", "pointwise", "cat",
                              "gelu", "add", "mul", "fill", "zero", "index")):
        return "elementwise / fusion"
    return "other"


def native_model(ours):
    """The textbook spelling, sharing our weights -- the control in the five-way table."""
    from torch_fno.fno.layers import add_grid, gelu

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


def kernel_table(f, x, label, case, batch, rows):
    """Device time and launch count per bucket, averaged over ten calls."""
    with torch.no_grad():
        for _ in range(5):
            f(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                f(x)
            torch.cuda.synchronize()
    totals, counts, grand, launches = {}, {}, 0.0, 0
    for e in prof.key_averages():
        us = e.self_device_time_total
        if us <= 0:
            continue
        key = bucket(e.key)
        totals[key] = totals.get(key, 0.0) + us
        counts[key] = counts.get(key, 0) + e.count
        grand += us
        launches += e.count
    print(f"\n  {label}: {grand / 10:9.1f} us/call device, {launches // 10} kernel launches",
          flush=True)
    for k in sorted(totals, key=lambda k: -totals[k]):
        print(f"      {k:22} {totals[k] / 10:9.1f} us {100 * totals[k] / grand:5.1f}%"
              f"   {counts[k] // 10:4d} launches", flush=True)
        rows.append({"case": case, "impl": label, "batch": batch, "bucket": k,
                     "us_per_call": totals[k] / 10, "pct": 100 * totals[k] / grand,
                     "launches": counts[k] // 10})
    top = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:8]
    print("      top kernels:", flush=True)
    for e in top:
        if e.self_device_time_total > 0:
            print(f"        {e.self_device_time_total / 10:8.1f} us  x{e.count // 10:3d}  "
                  f"{e.key[:70]}", flush=True)


def build_theirs(ch, their_m, n_layers=4, **kw):
    return TheirFNO(n_modes=(their_m, their_m), in_channels=1, out_channels=1,
                    hidden_channels=ch, n_layers=n_layers, **kw).to(DEV).eval()


# ---------------------------------------------------------------- A  ablation
print("=" * 78, flush=True)
print("A  which default costs the time   (us/sample, best of 20, shared batch)", flush=True)
print("=" * 78, flush=True)

ablate_rows = []
for name, shape, ch, modes, batch, their_m in CASES:
    x = torch.randn(batch, 1, *shape, device=DEV)
    print(f"\n{name}  grid {shape}  C={ch}  B={batch}", flush=True)
    base = None
    for label, kw in VARIANTS:
        try:
            mod = build_theirs(ch, their_m, **kw)
            us = timed(mod, x) / batch
            params = sum(p.numel() for p in mod.parameters())
            del mod
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"    {label:22} FAILED {type(exc).__name__} {str(exc)[:50]}", flush=True)
            continue
        if base is None:
            base = us
        print(f"    {label:22} {us:9.2f} us/sa   {base / us:5.2f}x faster than default"
              f"   params {params:,}", flush=True)
        ablate_rows.append({"case": name, "variant": label, "batch": batch,
                            "us_per_sample": us, "speedup_vs_default": base / us,
                            "params": params})
    del x
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- B  kernel trace
print("\n" + "=" * 78, flush=True)
print("B  what the device runs   (10-call average)", flush=True)
print("=" * 78, flush=True)

kern_rows = []
for name, shape, ch, modes, batch, their_m in CASES:
    x = torch.randn(batch, 1, *shape, device=DEV)
    print(f"\n{name}  B={batch}", flush=True)

    ours = OurFNO(1, 1, ch, modes, 4).to(DEV).eval()
    ours.load_state_dict(torch.load(f"/home/anirudh/{name}_state.pt", map_location="cpu"))
    ours = ours.to(DEV).eval()
    for b in ours.blocks:
        b.spectral.layout = "mode"
    ours.prebuild(torch.float32)

    kernel_table(native_model(ours), x, "native torch", name, batch, kern_rows)
    theirs = build_theirs(ch, their_m)
    kernel_table(theirs, x, "their torch (default)", name, batch, kern_rows)
    lean = build_theirs(ch, their_m, **dict(VARIANTS[-1][1]))
    kernel_table(lean, x, "their torch (four off)", name, batch, kern_rows)

    del ours, theirs, lean, x
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- C  its own batch
print("\n" + "=" * 78, flush=True)
print("C  is the shared batch unfair to it   (sweep its own batch)", flush=True)
print("=" * 78, flush=True)

LADDERS = {"Burgers": [64, 128, 256, 396, 512, 768, 1024],
           "Darcy-32": [32, 64, 128, 170, 256, 384],
           "Darcy-64": [8, 16, 32, 44, 64, 96],
           "NS-128": [1, 2, 4, 6, 8]}

batch_rows = []
for name, shape, ch, modes, shared_batch, their_m in CASES:
    print(f"\n{name}  (shared batch was {shared_batch})", flush=True)
    theirs = build_theirs(ch, their_m)
    best, best_b = float("inf"), None
    for b in LADDERS[name]:
        try:
            x = torch.randn(b, 1, *shape, device=DEV)
            us = timed(theirs, x, reps=10, warm=4) / b
            del x
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    B={b:5d}  out of memory", flush=True)
            continue
        mark = ""
        if us < best:
            best, best_b, mark = us, b, "  <- best"
        print(f"    B={b:5d}  {us:9.2f} us/sa{mark}", flush=True)
        batch_rows.append({"case": name, "batch": b, "us_per_sample": us,
                           "is_shared_batch": int(b == shared_batch)})
    shared = [r["us_per_sample"] for r in batch_rows
              if r["case"] == name and r["is_shared_batch"]]
    if shared:
        print(f"    best B={best_b} at {best:.2f};  shared B={shared_batch} at "
              f"{shared[0]:.2f}  ->  shared batch costs {shared[0] / best:.3f}x", flush=True)
    del theirs
    torch.cuda.empty_cache()

for rows, path in ((ablate_rows, "/home/anirudh/why_theirs_ablate.csv"),
                   (kern_rows, "/home/anirudh/why_theirs_kernels.csv"),
                   (batch_rows, "/home/anirudh/why_theirs_batch.csv")):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}", flush=True)
