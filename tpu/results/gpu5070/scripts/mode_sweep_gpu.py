"""Cost against the kept band on the GPU, for both torch spellings.

The TPU mode sweep covers three of the four datasets and reports accuracy beside cost. Accuracy
at a given band is a property of the problem and the trained weights, not of the device, so it
is not re-measured here; what is device-dependent is the cost, and that is what this sweeps.
Weights are random because cost does not depend on their values, and each band is timed at the
batch that band's own capacity bound suggests rather than at a batch fixed across the sweep.
"""
import csv
import sys

sys.path.insert(0, "/home/anirudh")

import torch

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune
from torch_fno.fno.layers import add_grid, gelu
from torch_fno.fno.model import FNO

DEV = torch.device("cuda")
ACCEL = detect()
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

# (case, grid shape, channels, the bands to sweep). The band counts both row corners and the
# column half, so it runs to the grid width; every even value is swept, and Navier-Stokes steps
# by four because its weight grows with the square of the band and leaves memory sooner.
CASES = [("Burgers", (17, 16), 24, list(range(2, 17, 2))),
         ("Darcy-32", (32, 32), 24, list(range(2, 33, 2))),
         ("Darcy-64", (64, 64), 24, list(range(2, 65, 2))),
         ("NS-128", (128, 128), 64, list(range(4, 129, 4)))]
REPS = 20


def timed(f, x, reps=REPS, warm=8):
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


def native_model(ours):
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


rows = []
for name, shape, ch, bands in CASES:
    print(f"\n{'=' * 74}\n{name}  grid {shape}  C={ch}\n{'=' * 74}", flush=True)
    print(f"{'modes':>6} {'batch':>6} {'params':>10} {'ours us/sa':>11} "
          f"{'native us/sa':>13} {'ratio':>7}", flush=True)
    for modes in bands:
        spec = ModelSpec(grid=max(shape), channels=ch, n_modes=modes)
        batch = AutoTune(spec, ACCEL, max_batch=1000).batch("float32")
        batch = batch[0] if isinstance(batch, tuple) else batch
        try:
            ours = FNO(1, 1, ch, modes, 4).to(DEV).eval()
            for b in ours.blocks:
                b.spectral.layout = "mode"
            ours.prebuild(torch.float32)
            x = torch.randn(batch, 1, *shape, device=DEV)
            mine = timed(ours, x) / batch
            native = timed(native_model(ours), x) / batch
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{modes:6d} {batch:6d}  out of memory", flush=True)
            continue
        params = ours.n_params()
        print(f"{modes:6d} {batch:6d} {params:10,} {mine:11.3f} {native:13.3f} "
              f"{mine / native:6.2f}x", flush=True)
        rows.append({"target": ACCEL.name, "case": name, "n_modes": modes, "batch": batch,
                     "params": params, "ours_us_per_sample": mine,
                     "native_us_per_sample": native, "ratio": mine / native})
        del ours, x
        torch.cuda.empty_cache()
        with open("/home/anirudh/fno/mode_sweep_gpu.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

with open("/home/anirudh/fno/mode_sweep_gpu.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote mode_sweep_gpu.csv ({len(rows)} rows)", flush=True)
