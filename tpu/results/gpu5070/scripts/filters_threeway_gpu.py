"""Where the time goes when the transform and the channel filter come from different places.

Two things are being changed independently: which transform computes the spectrum, and which
implementation mixes the channels. Timing only the end points cannot separate them, so this
runs three arms over the same three filters.

  native + native   torch.fft, and the filter as a complex einsum, which is how the published
                    implementations write it
  native + ours     torch.fft, and our filter: a real block matmul on real/imag pairs
  ours   + ours     our partial transform as well, the whole path

The middle arm is the one that isolates the filter, because it differs from the first only in
how the channel mix is issued. It is built by replacing one method pair on each spectral block,
so the filter code it runs is the same object the third arm runs -- not a reimplementation.

Correctness is checked before anything is timed: on the dense filter all three arms share
weights, so the first and second must agree to transform roundoff.
"""
import collections
import csv
import json
import os
import sys
import types

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune
from torch_fno.fno.layers import add_grid, gelu
from torch_fno.fno.model import FNO

DEV = torch.device("cuda")
ACCEL = detect()
OUT = os.environ.get("THREEWAY_OUT", "/home/anirudh/fno/filters_threeway_gpu.csv")
KERNELS = os.environ.get("THREEWAY_KERNELS", "/home/anirudh/fno/filters_threeway_kernels.json")
REPS, WARMUP, RANK, SPLIT = 30, 8, 8, 1000
CASES = [("Burgers", (17, 16), 24, 16), ("Darcy-32", (32, 32), 24, 16),
         ("Darcy-64", (64, 64), 24, 16), ("NS-128", (128, 128), 64, 64)]
FILTERS = ["dense", "ffno", "tfno"]
ARMS = ["native+native", "native+ours", "ours+ours"]
FIELDS = ["case", "grid", "channels", "n_modes", "filter", "arm", "batch",
          "us_per_sample", "params", "rel_err", "status"]

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True


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


# -- the hybrid arm ------------------------------------------------------------------------

def _fft_analyse(self, x):
    """(n, c, h, w) -> (a, d, n, 2c), the layout every `mix` expects, built by torch.fft."""
    rows, cols = self.bands(x)
    index = self._fft_index(rows, cols, x.device)
    spectrum = torch.fft.rfft2(x, dim=(-2, -1))
    band = spectrum[:, :, index[0]][:, :, :, index[1]]
    return torch.cat([band.real.permute(2, 3, 0, 1),
                      band.imag.permute(2, 3, 0, 1)], dim=3).to(self.dtype)


def _fft_synthesise(self, mixed, n_ch, x_shape):
    """(a, d, n, 2o) -> (n, o, h, w). irfft2 already carries the 1/(hw), so none is applied."""
    n, _, h, w = x_shape
    rows, cols = self.bands_cached
    index = self._fft_index(rows, cols, mixed.device)
    out = torch.zeros(n, n_ch, h, w // 2 + 1, dtype=torch.complex64, device=mixed.device)
    band = torch.complex(mixed[..., :n_ch].float(), mixed[..., n_ch:].float())
    out[:, :, index[0][:, None], index[1][None, :]] = band.permute(2, 3, 0, 1)
    return torch.fft.irfft2(out, s=(h, w), dim=(-2, -1)).to(torch.float32)


def _fft_index(self, rows, cols, device):
    key = (len(rows), len(cols), str(device))
    cached = getattr(self, "_idx_cache", None)
    if cached is None or cached[0] != key:
        self._idx_cache = (key,
                           (torch.as_tensor(rows, dtype=torch.long, device=device),
                            torch.as_tensor(cols, dtype=torch.long, device=device)))
    return self._idx_cache[1]


def make_hybrid(model):
    """Swap each block's transform for torch.fft and leave its filter alone."""
    for block in model.blocks:
        spectral = block.spectral
        spectral.layout = "mode"
        spectral._fft_index = types.MethodType(_fft_index, spectral)
        spectral.analyse_mode_major = types.MethodType(_fft_analyse, spectral)
        spectral.synthesise_mode_major = types.MethodType(_fft_synthesise, spectral)
    return model


# -- the published arm ---------------------------------------------------------------------

def native_runner(ours, which, channels, n_modes):
    """torch.fft plus the filter as a complex einsum, the way the papers write it."""
    half = n_modes // 2
    cdt = torch.complex64

    def randc(*shape):
        return (torch.randn(*shape, device=DEV) + 1j * torch.randn(*shape, device=DEV)).to(cdt)

    if which == "dense":
        weights = [torch.complex(b.spectral.w_re, b.spectral.w_im) for b in ours.blocks]
    elif which == "ffno":
        weights = [(randc(n_modes, channels, channels), randc(half, channels, channels))
                   for _ in ours.blocks]
    else:
        weights = [(randc(channels, RANK), randc(channels, RANK),
                    randc(n_modes, RANK), randc(half, RANK)) for _ in ours.blocks]

    def mix(w, lo, hi):
        if which == "dense":
            return (torch.einsum("adio,niad->noad", w[:half], lo),
                    torch.einsum("adio,niad->noad", w[half:], hi))
        if which == "ffno":
            k, l = w
            return tuple(torch.einsum("aio,niad->noad", krows, part)
                         + torch.einsum("dio,niad->noad", l, part)
                         for part, krows in ((lo, k[:half]), (hi, k[half:])))
        a, b, c, d = w
        out = []
        for part, crows in ((lo, c[:half]), (hi, c[half:])):
            v = torch.einsum("ir,niad->nrad", a, part)
            v = v * (crows.T[:, :, None] * d.T[:, None, :]).unsqueeze(0)
            out.append(torch.einsum("or,nrad->noad", b, v))
        return tuple(out)

    def run(x):
        x = ours.lift(add_grid(x))
        for blk, w in zip(ours.blocks, weights):
            n, c, h, wid = x.shape
            xf = torch.fft.rfft2(x, dim=(-2, -1))
            lo, hi = xf[:, :, :half, :half], xf[:, :, -half:, :half]
            mixed_lo, mixed_hi = mix(w, lo, hi)
            out = torch.zeros(n, c, h, wid // 2 + 1, dtype=xf.dtype, device=x.device)
            out[:, :, :half, :half] = mixed_lo
            out[:, :, h - half:, :half] = mixed_hi
            x = gelu(torch.fft.irfft2(out, s=(h, wid), dim=(-2, -1)) + blk.skip(x))
        return ours.project(x)
    return run


def agreement(a, b, shape):
    x = torch.randn(2, 1, *shape, device=DEV)
    with torch.no_grad():
        u, v = a(x), b(x)
    return float((u - v).abs().max() / u.abs().max())


def kernel_tally(fn, x):
    """Which CUDA kernels one call issues, so 'does it reach the tensor cores' has an answer."""
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        for _ in range(3):
            fn(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(x)
            torch.cuda.synchronize()
    totals = collections.Counter()
    for event in prof.key_averages():
        if event.self_device_time_total > 0 and event.key.startswith(("void ", "nvjet", "ampere",
                                                                     "sm80", "cutlass", "vector")):
            totals[event.key[:110]] += event.self_device_time_total
    return totals


def main():
    rows_out, kernels, checks = [], {}, {}
    for case, shape, channels, n_modes in CASES:
        spec = ModelSpec(grid=max(shape), channels=channels, n_modes=n_modes)
        batch = AutoTune(spec, ACCEL, max_batch=SPLIT).batch("float32")
        batch = batch[0] if isinstance(batch, tuple) else batch
        print(f"\n{case}: grid {shape}, channels {channels}, n_modes {n_modes}, batch {batch}",
              flush=True)

        for which in FILTERS:
            torch.manual_seed(0)
            ours = FNO(1, 1, channels, n_modes, 4, which).to(DEV).eval()
            ours.prebuild(torch.float32)
            torch.manual_seed(0)
            hybrid = make_hybrid(FNO(1, 1, channels, n_modes, 4, which).to(DEV).eval())
            hybrid.prebuild(torch.float32)
            native = native_runner(ours, which, channels, n_modes)

            # The hybrid and ours share weights and differ only in the transform, so on any
            # filter they must agree; the dense arm additionally matches the published one.
            checks[(case, which)] = {"hybrid_vs_ours": agreement(hybrid, ours, shape)}
            if which == "dense":
                checks[(case, which)]["native_vs_ours"] = agreement(native, ours, shape)

            x = torch.randn(batch, 1, *shape, device=DEV)
            for arm, fn in (("native+native", native), ("native+ours", hybrid),
                            ("ours+ours", ours)):
                row = {"case": case, "grid": max(shape), "channels": channels,
                       "n_modes": n_modes, "filter": which, "arm": arm, "batch": batch,
                       "params": ours.n_params() if arm != "native+native" else "",
                       "rel_err": f"{checks[(case, which)].get('hybrid_vs_ours', 0):.3e}"
                                  if arm == "native+ours" else ""}
                try:
                    row["us_per_sample"] = f"{timed(fn, x) / batch:.4f}"
                    row["status"] = "ok"
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    row.update({"us_per_sample": "", "status": "oom"})
                print(f"  {which:6} {arm:14} {row['us_per_sample'] or 'oom':>10} us/sample",
                      flush=True)
                rows_out.append(row)
                if case == "NS-128" and row["status"] == "ok":
                    kernels[f"{which}|{arm}"] = dict(kernel_tally(fn, x))
            del x, ours, hybrid, native
            torch.cuda.empty_cache()

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    with open(KERNELS, "w") as fh:
        json.dump(kernels, fh, indent=1)
    print(f"\nwrote {OUT} ({len(rows_out)} rows) and {KERNELS}", flush=True)
    print("\nagreement (max rel err against our end-to-end path):", flush=True)
    for (case, which), got in checks.items():
        print(f"  {case:9} {which:6} " + "  ".join(f"{k}={v:.2e}" for k, v in got.items()),
              flush=True)
    print("\nTHREEWAY_DONE", flush=True)


if __name__ == "__main__":
    main()
