"""The same contraction, the weight stored two ways.

The reference implementation keeps its filter channel-major, `(in, out, row, col)`, and
contracts with `nixy,ioxy->noxy`.  A batched matmul wants the mode axes leading, so the whole
weight is transposed into `(row, col, in, out)` on every call.  Our filter is stored in that
order already -- the `adio` legend -- so the transpose never happens.
"""
import sys

sys.path.insert(0, "/home/anirudh")
sys.path.insert(0, "/home/anirudh/fno/nop")

import torch

DEV = torch.device("cuda")

# (case, batch, channels, kept rows, kept cols)
CASES = [("Darcy-64", 32, 24, 15, 8), ("NS-128", 4, 64, 63, 32)]


def timed(f, reps=30, warm=10):
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


print(f"{'case':10} {'weight MiB':>11} {'channel-major':>14} {'mode-major':>12} "
      f"{'gain':>6} {'x4 saved':>10}")
print("-" * 72)

with torch.no_grad():
    for name, batch, ch, rows, cols in CASES:
        band = torch.randn(batch, ch, rows, cols, dtype=torch.cfloat, device=DEV)

        # Their storage order, and the same numbers stored the way the matmul wants them.
        w_chan = torch.randn(ch, ch, rows, cols, dtype=torch.cfloat, device=DEV)
        w_mode = w_chan.permute(2, 3, 0, 1).contiguous()
        mib = w_chan.numel() * 8 / 2**20

        theirs = timed(lambda: torch.einsum("nixy,ioxy->noxy", band, w_chan))
        ours = timed(lambda: torch.einsum("nixy,xyio->noxy", band, w_mode))

        agree = torch.allclose(torch.einsum("nixy,ioxy->noxy", band, w_chan),
                               torch.einsum("nixy,xyio->noxy", band, w_mode), atol=1e-4)
        print(f"{name:10} {mib:9.1f} {theirs:12.1f} us {ours:9.1f} us "
              f"{theirs / ours:5.2f}x {4 * (theirs - ours):8.1f} us   "
              f"same result {agree}")
        del band, w_chan, w_mode
        torch.cuda.empty_cache()

print("\nL2 on this part is 48 MiB, so the NS weight does not fit and the transpose "
      "goes to DRAM.")
