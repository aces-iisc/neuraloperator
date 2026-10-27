"""Walk the reference spectral convolution one step at a time, pricing each step.

The steps follow neuralop/layers/spectral_convolution.py: forward transform, centre the
spectrum, slice the kept band, scatter it back into a zero spectrum, un-centre, then the
split inverse that enforces Hermitian symmetry.  Our filter does none of the centring and
none of the splitting, so the point is to see which of those steps the time sits in.
"""
import sys

sys.path.insert(0, "/home/anirudh")
sys.path.insert(0, "/home/anirudh/fno/nop")

import torch

DEV = torch.device("cuda")
N_LAYERS = 4

# (case, batch, channels, grid, their m)
CASES = [("Darcy-64", 32, 24, 64, 15), ("NS-128", 4, 64, 128, 63)]


def timed(f, reps=30, warm=8):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(reps):
        begin.record()
        f()
        end.record()
        torch.cuda.synchronize()
        best = min(best, begin.elapsed_time(end) * 1e3)
    return best


with torch.no_grad():
    for name, batch, ch, grid, m in CASES:
        field = torch.randn(batch, ch, grid, grid, device=DEV)
        cols = grid // 2 + 1
        spectrum = torch.fft.rfftn(field, dim=(-2, -1))
        centred = torch.fft.fftshift(spectrum, dim=(-2,))

        keep_cols = m // 2 + 1
        centre, neg, pos = grid // 2, m // 2, m // 2 + m % 2
        row_slice = slice(centre - neg, centre + pos)
        band = centred[:, :, row_slice, :keep_cols].contiguous()
        out_fft = torch.zeros(batch, ch, grid, cols, dtype=torch.cfloat, device=DEV)

        print("=" * 76, flush=True)
        print(f"{name}  B={batch}  C={ch}  grid {grid}  m={m}   "
              f"spectrum {tuple(spectrum.shape)}  band {tuple(band.shape)}", flush=True)
        print(f"{'step':40} {'us/call':>9} {'x4 layers':>11}", flush=True)
        print("-" * 76, flush=True)

        steps = [
            ("rfftn (forward transform)",
             lambda: torch.fft.rfftn(field, dim=(-2, -1))),
            ("fftshift (centre the spectrum)",
             lambda: torch.fft.fftshift(spectrum, dim=(-2,))),
            ("slice the kept band, contiguous",
             lambda: centred[:, :, row_slice, :keep_cols].contiguous()),
            ("zeros for the full spectrum",
             lambda: torch.zeros(batch, ch, grid, cols, dtype=torch.cfloat, device=DEV)),
            ("scatter the band back in",
             lambda: out_fft[:, :, row_slice, :keep_cols].copy_(band)),
            ("ifftshift (un-centre)",
             lambda: torch.fft.ifftshift(out_fft, dim=(-2,))),
            ("ifftn over rows (split inverse)",
             lambda: torch.fft.ifftn(out_fft, s=(grid,), dim=(-2,))),
            ("irfft over cols (split inverse)",
             lambda: torch.fft.irfft(out_fft, n=grid, dim=-1)),
            ("irfftn in one call (what we do)",
             lambda: torch.fft.irfftn(out_fft, s=(grid, grid), dim=(-2, -1))),
        ]

        total_shift = 0.0
        for label, fn in steps:
            us = timed(fn)
            print(f"{label:40} {us:9.1f} {us * N_LAYERS:11.1f}", flush=True)
            if "shift" in label:
                total_shift += us

        # The two Hermitian fixups, on the row-inverted spectrum.
        rowwise = torch.fft.ifftn(out_fft, s=(grid,), dim=(-2,))

        def hermitian():
            rowwise[..., 0].imag.zero_()
            rowwise[..., -1].imag.zero_()

        us = timed(hermitian)
        print(f"{'two imag.zero_() fixups':40} {us:9.1f} {us * N_LAYERS:11.1f}", flush=True)
        print(f"\n  centring cost alone: {total_shift * N_LAYERS:.1f} us over 4 layers",
              flush=True)
        del field, spectrum, centred, band, out_fft, rowwise
        torch.cuda.empty_cache()
        print(flush=True)
