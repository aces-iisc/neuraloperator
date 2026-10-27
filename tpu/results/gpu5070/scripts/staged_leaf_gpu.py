"""Why a packing scheme ever wins: the staged plan's leaf, and the kernels it issues.

A radix-R staged transform peels one factor of R off the length per stage and finishes with a
direct DFT on whatever is left. That remainder -- the leaf -- is the whole story: a leaf of 2
occupies 4 cells of a tile that is applied to every column of the batch regardless, so it runs
at a tiny fraction of the rate a full leaf does. The reduced scheme transforms N/2 instead of N,
which changes the remainder, and that is a saving no amount of band narrowing can reach.

So this prints the plan each engine actually gets at every N, and splits the measured time into
the matrix unit and everything else. The split is what decides the outcome: the leaf argument is
about matmul time alone, and the packing schemes pay a fixed unpack cost that the saving has to
clear before it becomes a win.
"""
import os
import sys

sys.path.insert(0, "/home/anirudh/fno")

import numpy as np
import torch

from torch_fno.fft.core import RADIX
from torch_fno.fft.rfft1d import rfft_half
from torch_fno.fft.tricks import rfft_coupled_half, rfft_reduced_half

DEV = torch.device("cuda")
GRIDS = [2 ** p for p in range(4, 13)]
PROFILE = [int(n) for n in os.environ.get("LEAF_PROFILE", "256,512,1024").split(",") if n]
K = int(os.environ.get("LEAF_K", "256"))


def plan(length, radix=RADIX):
    """(stages, leaf) for the staged transform of this length, read off the same loop."""
    stages = 0
    while length > radix and length % radix == 0:
        length //= radix
        stages += 1
    return stages, length


def kernels(fn, x, reps=20):
    """{kernel name: total us} over `reps` calls, device side only."""
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad():
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(reps):
                fn(x)
            torch.cuda.synchronize()
    totals = {}
    for event in prof.key_averages():
        if event.self_device_time_total:
            totals[event.key] = event.self_device_time_total / reps
    return totals


def gemm_split(totals):
    """(matrix-unit us, everything else us) for one engine's kernel tally."""
    mm = sum(us for key, us in totals.items()
             if "gemm" in key or "cutlass" in key or "splitKreduce" in key)
    return mm, sum(totals.values()) - mm


def main():
    print(f"radix {RADIX}\n")
    print(f"{'N':>6} {'reim plan':>18} {'reduced plan (N/2)':>22} {'coupled plan':>18}"
          f"   predicted")
    for grid in GRIDS:
        s_r, leaf_r = plan(grid)
        s_d, leaf_d = plan(grid // 2)
        verdict = ("reduced: full leaf against a starved one" if leaf_d > leaf_r
                   else "no leaf to gain")
        print(f"{grid:>6} {f'{s_r} stages, leaf {leaf_r}':>18} "
              f"{f'{s_d} stages, leaf {leaf_d}':>22} {f'{s_r} stages, leaf {leaf_r}':>18}"
              f"   {verdict}")

    rng = np.random.default_rng(0)
    print(f"\n{'N':>6}   " + "".join(f"{e:>24}" for e in ("reim", "coupled", "reduced")))
    print(f"{'':>6}   " + "".join(f"{'matmul':>10}{'other':>8}{'tot':>6}" for _ in range(3)))
    for grid in GRIDS:
        x = torch.as_tensor(rng.normal(size=(grid, K)).astype(np.float32), device=DEV)
        cells = []
        for fn in (rfft_half, rfft_coupled_half, rfft_reduced_half):
            mm, rest = gemm_split(kernels(fn, x))
            cells.append(f"{mm:10.1f}{rest:8.1f}{mm + rest:6.0f}")
        print(f"{grid:>6}   " + "".join(cells), flush=True)
        del x
        torch.cuda.empty_cache()

    for grid in PROFILE:
        x = torch.as_tensor(rng.normal(size=(grid, K)).astype(np.float32), device=DEV)
        print(f"\n--- N={grid}, K={K}: top kernels, us per call ---")
        for name, fn in (("reim", rfft_half), ("coupled", rfft_coupled_half),
                         ("reduced", rfft_reduced_half)):
            totals = kernels(fn, x)
            ordered = sorted(totals.items(), key=lambda kv: -kv[1])
            print(f"  {name:8} total {sum(totals.values()):8.2f}  "
                  f"({len(totals)} distinct kernels)")
            for key, us in ordered[:5]:
                print(f"      {us:8.2f}  {key[:66]}")
        del x
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
