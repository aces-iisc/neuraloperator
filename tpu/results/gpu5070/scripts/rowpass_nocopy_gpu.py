"""Three ways to apply the row DFT to a field set, measured for time and for peak memory.

The row pass contracts the h axis of an (n, c, h, w) field set against a (2m, h) DFT matrix.
How the operand is presented to the GEMM decides whether a copy of the whole field set has to
exist at the same time as the field set itself:

  relayout  x.permute(2, 0, 1, 3).reshape(h, n*c*w), then one GEMM of width n*c*w.  The permute
            leaves the tensor non-contiguous, so the reshape materialises a second copy of
            every element.  One wide GEMM, peak memory about twice the input.
  batched   x.reshape(n*c, h, w) is a view, and matmul broadcasts the (2m, h) matrix over it.
            n*c strided GEMMs of width w, no copy at all.
  einsum    the same contraction written as an einsum, leaving the choice to the backend.

The three agree to float32 round-off, so the question is only cost.  Peak memory is read from
the caching allocator, reset per measurement, and is the figure that decides how large a field
set the sweep can reach at all.
"""
import csv
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import torch

from torch_fno.fft.core import ACC, F, stacked_rows

DEV = torch.device("cuda")
OUT = os.environ.get("NOCOPY_OUT", "/home/anirudh/fno/rowpass_nocopy_gpu.csv")
REPS = int(os.environ.get("NOCOPY_REPS", 20))
WARMUP = 5
BUDGET = float(os.environ.get("NOCOPY_BUDGET", 3.0)) * 2 ** 30
FIELDS = ["grid", "B", "n_modes", "method", "us", "peak_mib", "input_mib", "rel_err", "status"]

# (N, B) pairs that span the regimes: wide-and-shallow, square, and tall-and-narrow.
CASES = [(64, 2000), (64, 20000), (128, 500), (128, 5000), (256, 100), (256, 1000),
         (512, 50), (512, 500), (1024, 20), (1024, 200), (2048, 10), (2048, 50),
         (4096, 1), (4096, 10), (4096, 20)]


def relayout(matrix, x, m):
    n, c, h, w = x.shape
    flat = x.permute(2, 0, 1, 3).reshape(h, n * c * w)
    both = torch.matmul(matrix, flat).to(ACC)
    return both[:m].reshape(m, n, c, w).permute(1, 2, 0, 3)


def batched(matrix, x, m):
    n, c, h, w = x.shape
    both = torch.matmul(matrix, x.reshape(n * c, h, w)).to(ACC)
    return both[:, :m].reshape(n, c, m, w)


def einsum(matrix, x, m):
    both = torch.einsum("kh,nchw->nckw", matrix, x).to(ACC)
    return both[:, :, :m]


METHODS = [("relayout", relayout), ("batched", batched), ("einsum", einsum)]


def timed(fn):
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.no_grad():
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        best = float("inf")
        for _ in range(REPS):
            begin.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def peak_of(fn):
    """Peak bytes the allocator held during one call, the input excluded from the reset."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.no_grad():
        out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del out
    return peak, before


def main():
    gen = torch.Generator(device=DEV).manual_seed(0)
    rows = []
    blank = {f: "" for f in FIELDS}
    print(f"{'N':>6}{'B':>8}{'m':>6}{'method':>10}{'us':>11}{'peak MiB':>11}"
          f"{'in MiB':>9}{'rel err':>11}", flush=True)
    for grid, batch in CASES:
        want = batch * grid * grid * 4
        if want > BUDGET:
            print(f"  N={grid} B={batch} skipped, {want / 2 ** 30:.1f} GiB", flush=True)
            continue
        x = torch.randn((batch, 1, grid, grid), device=DEV, generator=gen, dtype=torch.float32)
        for m in sorted({max(2, grid // 8), grid // 2}):
            matrix = stacked_rows(range(m), grid, -1, F, DEV)
            reference = None
            for name, fn in METHODS:
                row = dict(blank, grid=grid, B=batch, n_modes=m, method=name,
                           input_mib=f"{want / 2 ** 20:.1f}")
                call = lambda _f=fn, _m=m: _f(matrix, x, _m)
                try:
                    got = call()
                    if reference is None:
                        reference = got.float()
                        row["rel_err"] = "0.000e+00"
                    else:
                        scale = reference.abs().max().clamp_min(1e-30)
                        row["rel_err"] = (f"{((got.float() - reference).abs().max() / scale):.3e}")
                    del got
                    peak, before = peak_of(call)
                    row["peak_mib"] = f"{(peak - before) / 2 ** 20:.1f}"
                    row["us"] = f"{timed(call):.2f}"
                    row["status"] = "ok"
                except Exception as exc:
                    row["status"] = ("OOM" if "out of memory" in str(exc).lower()
                                     else type(exc).__name__)
                    torch.cuda.empty_cache()
                rows.append(row)
                print(f"{grid:>6}{batch:>8}{m:>6}{name:>10}{row['us'] or '-':>11}"
                      f"{row['peak_mib'] or '-':>11}{row['input_mib']:>9}"
                      f"{row['rel_err'] or '-':>11}  {row['status']}", flush=True)
            del matrix, reference
            torch.cuda.empty_cache()
        del x
        torch.cuda.empty_cache()
        with open(OUT, "w", newline="") as fh:
            writer = csv.DictWriter(fh, FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    print("NOCOPY_DONE", flush=True)


if __name__ == "__main__":
    main()
