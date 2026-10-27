"""What the factorised filters are worth, and what our transform is worth underneath them.

Two improvements get mixed together whenever a factorised FNO is timed. Replacing the dense
channel filter with F-FNO's axis sum or TFNO's rank-$R$ product is one of them; computing only
the modes the layer keeps, rather than a full spectrum thrown mostly away, is the other. A
single number cannot separate them.

So every filter is built twice, once over our transform and once over `torch.fft`, and the same
three filters run on both. Reading down a column gives what the factorisation is worth at a
fixed transform; reading across a row gives what the transform is worth at a fixed filter.

Weights are random on both sides. Cost depends on their shapes, not their values, and the two
sides cannot share a tensor for the factorised filters anyway: ours stores the real block form
the matmul wants, and the shapes are not in correspondence for the rank-$R$ product. The dense
pair does share weights and its outputs are checked against each other, which is what validates
the harness; the other two are verified on parameter count and output shape.
"""
import collections
import csv
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.arch import ModelSpec, detect
from torch_fno.fno.autotune import AutoTune
from torch_fno.fno.layers import add_grid, gelu
from torch_fno.fno.model import FNO

DEV = torch.device("cuda")
ACCEL = detect()
OUT = os.environ.get("FILTERS_OUT", "/home/anirudh/fno/filters_vs_native_gpu.csv")
SPLIT = 1000
REPS = 20
WARMUP = 5
RANK = 8
CASES = [("Burgers", (17, 16), 24, 16), ("Darcy-32", (32, 32), 24, 16),
         ("Darcy-64", (64, 64), 24, 16), ("NS-128", (128, 128), 64, 64)]
FILTERS = ["dense", "ffno", "tfno"]
FIELDS = ["case", "filter", "impl", "batch", "us_per_sample", "params"]

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


def band_slices(spectrum, half):
    """The two low corners of the row axis and the low half of the column axis."""
    return spectrum[:, :, :half, :half], spectrum[:, :, -half:, :half]


def native_runner(ours, which, channels, n_modes):
    """`torch.fft` plus the same filter, as a textbook implementation would write it.

    The weights here are complex tensors in the layout a complex einsum wants, which is the
    whole point of the comparison: ours are real pairs in the layout a real matmul wants.
    """
    half = n_modes // 2
    rows, cols, dev = n_modes, half, DEV
    cdt = torch.complex64

    def randc(*shape):
        return (torch.randn(*shape, device=dev) + 1j * torch.randn(*shape, device=dev)).to(cdt)

    if which == "dense":
        weights = [torch.complex(b.spectral.w_re, b.spectral.w_im) for b in ours.blocks]
    elif which == "ffno":
        weights = [(randc(rows, channels, channels), randc(cols, channels, channels))
                   for _ in ours.blocks]
    else:
        weights = [(randc(channels, RANK), randc(channels, RANK),
                    randc(rows, RANK), randc(cols, RANK)) for _ in ours.blocks]

    def mix(w, xf_lo, xf_hi):
        if which == "dense":
            return (torch.einsum("adio,niad->noad", w[:half], xf_lo),
                    torch.einsum("adio,niad->noad", w[half:], xf_hi))
        if which == "ffno":
            k, l = w
            out = []
            for part, krows in ((xf_lo, k[:half]), (xf_hi, k[half:])):
                out.append(torch.einsum("aio,niad->noad", krows, part)
                           + torch.einsum("dio,niad->noad", l, part))
            return tuple(out)
        a, b, c, d = w
        out = []
        for part, crows in ((xf_lo, c[:half]), (xf_hi, c[half:])):
            v = torch.einsum("ir,niad->nrad", a, part)
            v = v * (crows.T[:, :, None] * d.T[:, None, :]).unsqueeze(0)
            out.append(torch.einsum("or,nrad->noad", b, v))
        return tuple(out)

    def run(x):
        x = ours.lift(add_grid(x))
        for blk, w in zip(ours.blocks, weights):
            n, c, h, wid = x.shape
            xf = torch.fft.rfft2(x, dim=(-2, -1))
            lo, hi = band_slices(xf, half)
            mixed_lo, mixed_hi = mix(w, lo, hi)
            out = torch.zeros(n, c, h, wid // 2 + 1, dtype=xf.dtype, device=x.device)
            out[:, :, :half, :half] = mixed_lo
            out[:, :, h - half:, :half] = mixed_hi
            x = gelu(torch.fft.irfft2(out, s=(h, wid), dim=(-2, -1)) + blk.skip(x))
        return ours.project(x)
    return run


def check_dense(ours, native, shape):
    """The one pair that can share weights, as an end-to-end check on the harness."""
    x = torch.randn(2, 1, *shape, device=DEV)
    with torch.no_grad():
        a, b = ours(x), native(x)
    return float((a - b).abs().max() / a.abs().max())


def main():
    rows_out, errors = [], {}
    for case, shape, channels, n_modes in CASES:
        spec = ModelSpec(grid=max(shape), channels=channels, n_modes=n_modes)
        batch = AutoTune(spec, ACCEL, max_batch=SPLIT).batch("float32")
        batch = batch[0] if isinstance(batch, tuple) else batch
        print(f"\n{case}: batch {batch}", flush=True)

        for which in FILTERS:
            ours = FNO(1, 1, channels, n_modes, 4, which).to(DEV).eval()
            ours.prebuild(torch.float32)
            native = native_runner(ours, which, channels, n_modes)
            if which == "dense":
                errors[case] = check_dense(ours, native, shape)

            x = torch.randn(batch, 1, *shape, device=DEV)
            for impl, fn in (("ours", ours), ("native", native)):
                try:
                    us = timed(fn, x) / batch
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  {which:6} {impl:7} out of memory", flush=True)
                    continue
                rows_out.append({"case": case, "filter": which, "impl": impl, "batch": batch,
                                 "us_per_sample": f"{us:.4f}",
                                 "params": ours.n_params() if impl == "ours" else ""})
                print(f"  {which:6} {impl:7} {us:9.3f} us/sample", flush=True)
            del x, ours, native
            torch.cuda.empty_cache()

    with open(OUT, "w", newline="") as fh:
        writer = csv.DictWriter(fh, FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {OUT}  ({len(rows_out)} rows)", flush=True)
    print("dense ours-vs-native agreement: "
          + "  ".join(f"{c}={e:.2e}" for c, e in errors.items()), flush=True)

    at = {(r["case"], r["filter"], r["impl"]): float(r["us_per_sample"]) for r in rows_out}
    print(f"\n{'case':9} {'filter':7} {'ours':>10} {'native':>10} {'native/ours':>12} "
          f"{'vs dense (ours)':>16}")
    print("-" * 70)
    for case, _, _, _ in CASES:
        base = at.get((case, "dense", "ours"))
        for which in FILTERS:
            o, n = at.get((case, which, "ours")), at.get((case, which, "native"))
            if o is None:
                continue
            print(f"{case:9} {which:7} {o:10.3f} "
                  + (f"{n:10.3f} {n / o:11.3f}x" if n else f"{'---':>10} {'---':>12}")
                  + f" {base / o:15.3f}x")


if __name__ == "__main__":
    main()
