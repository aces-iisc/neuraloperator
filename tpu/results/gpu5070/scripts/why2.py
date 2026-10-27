"""Where the reference implementation's time actually goes, once the band is matched.

  A  re-time it at a band matched to ours, instead of matched by `numel`
  B  kernel buckets and launch counts for each implementation
  C  sweep its own batch, in case the shared batch was the handicap
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

WEIGHTS = "/home/anirudh/fno"

# (case, grid, channels, our n_modes, batch, m as run, m band-matched)
CASES = [("Burgers", (17, 16), 24, 16, 396, 22, 15),
         ("Darcy-32", (32, 32), 24, 16, 128, 22, 15),
         ("Darcy-64", (64, 64), 24, 16, 32, 22, 15),
         ("NS-128", (128, 128), 64, 64, 4, 78, 63)]

LEAN = {"use_channel_mlp": False, "channel_mlp_skip": "identity",
        "enforce_hermitian_symmetry": False, "positional_embedding": None}


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
    if any(k in low for k in ("fft", "postprocess", "preprocess")):
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


def real_params(mod):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in mod.parameters())


def native_model(ours):
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


def build_ours(name, ch, modes):
    mod = OurFNO(1, 1, ch, modes, 4)
    mod.load_state_dict(torch.load(f"{WEIGHTS}/{name}_state.pt", map_location="cpu"))
    mod = mod.to(DEV).eval()
    for b in mod.blocks:
        b.spectral.layout = "mode"
    mod.prebuild(torch.float32)
    return mod


def build_theirs(ch, m, **kw):
    return TheirFNO(n_modes=(m, m), in_channels=1, out_channels=1, hidden_channels=ch,
                    n_layers=4, **kw).to(DEV).eval()


def kernel_table(f, x, label, case, batch, rows):
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
    print(f"\n  {label:30} {grand / 10:9.1f} us/call device   {launches // 10:4d} launches",
          flush=True)
    for k in sorted(totals, key=lambda k: -totals[k]):
        print(f"      {k:22} {totals[k] / 10:9.1f} us {100 * totals[k] / grand:5.1f}%"
              f"   {counts[k] // 10:4d} launches", flush=True)
        rows.append({"case": case, "impl": label, "batch": batch, "bucket": k,
                     "us_per_call": totals[k] / 10, "pct": 100 * totals[k] / grand,
                     "launches": counts[k] // 10})
    top = sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:6]
    for e in top:
        if e.self_device_time_total > 0:
            print(f"        {e.self_device_time_total / 10:8.1f} us x{e.count // 10:3d}  "
                  f"{e.key[:66]}", flush=True)


# ------------------------------------------------------------------ A  matched band
print("=" * 86, flush=True)
print("A  re-timed at a band matched to ours   (us/sample, best of 20, shared batch)",
      flush=True)
print("=" * 86, flush=True)

rows_a = []
for name, shape, ch, modes, batch, m_run, m_fair in CASES:
    x = torch.randn(batch, 1, *shape, device=DEV)
    ours = build_ours(name, ch, modes)
    native = native_model(ours)
    our_p = real_params(ours)
    print(f"\n{name}  grid {shape}  C={ch}  B={batch}", flush=True)

    entries = [("ours torch", ours, our_p), ("native torch", native, our_p)]
    for label, m, kw in ((f"their torch, m={m_run} (as published)", m_run, {}),
                         (f"their torch, m={m_fair} (band-matched)", m_fair, {}),
                         (f"their torch, m={m_fair} + four defaults off", m_fair, LEAN)):
        mod = build_theirs(ch, m, **kw)
        entries.append((label, mod, real_params(mod)))

    base = None
    for label, f, params in entries:
        us = timed(f, x) / batch
        if label == "native torch":
            base = us
        print(f"    {label:40} {us:9.2f} us/sa   real params {params:,}", flush=True)
        rows_a.append({"case": name, "impl": label, "batch": batch, "us_per_sample": us,
                       "real_params": params})
    for r in rows_a:
        if r["case"] == name and base:
            r["x_vs_native"] = r["us_per_sample"] / base

    for label, f, _ in entries[2:]:
        del f
    del ours, native, x
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ B  kernel trace
print("\n" + "=" * 86, flush=True)
print("B  what the device runs   (10-call average)", flush=True)
print("=" * 86, flush=True)

rows_b = []
for name, shape, ch, modes, batch, m_run, m_fair in CASES:
    x = torch.randn(batch, 1, *shape, device=DEV)
    print(f"\n{name}  B={batch}", flush=True)
    ours = build_ours(name, ch, modes)
    kernel_table(ours, x, "ours torch", name, batch, rows_b)
    kernel_table(native_model(ours), x, "native torch", name, batch, rows_b)
    pub = build_theirs(ch, m_run)
    kernel_table(pub, x, f"their torch m={m_run} (published)", name, batch, rows_b)
    fair = build_theirs(ch, m_fair)
    kernel_table(fair, x, f"their torch m={m_fair} (band-matched)", name, batch, rows_b)
    del ours, pub, fair, x
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ C  its own batch
print("\n" + "=" * 86, flush=True)
print("C  sweep its own batch, band-matched", flush=True)
print("=" * 86, flush=True)

LADDERS = {"Burgers": [64, 128, 256, 396, 512, 768, 1024],
           "Darcy-32": [32, 64, 128, 170, 256, 384],
           "Darcy-64": [8, 16, 32, 44, 64, 96],
           "NS-128": [1, 2, 4, 6, 8, 12]}

rows_c = []
for name, shape, ch, modes, shared, m_run, m_fair in CASES:
    print(f"\n{name}  m={m_fair}  (shared batch was {shared})", flush=True)
    theirs = build_theirs(ch, m_fair)
    best, best_b, at_shared = float("inf"), None, None
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
        if b == shared:
            at_shared = us
        print(f"    B={b:5d}  {us:9.2f} us/sa{mark}", flush=True)
        rows_c.append({"case": name, "m": m_fair, "batch": b, "us_per_sample": us,
                       "is_shared_batch": int(b == shared)})
    if at_shared:
        print(f"    best B={best_b} at {best:.2f};  shared B={shared} at {at_shared:.2f}"
              f"  ->  shared batch costs {at_shared / best:.3f}x", flush=True)
    del theirs
    torch.cuda.empty_cache()

for rows, path in ((rows_a, "/home/anirudh/fno/why2_matched.csv"),
                   (rows_b, "/home/anirudh/fno/why2_kernels.csv"),
                   (rows_c, "/home/anirudh/fno/why2_batch.csv")):
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path}", flush=True)
