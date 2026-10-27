"""Count real-valued parameters and mode cells on both sides, and find the honest match.

Our filter stores real and imaginary parts as two real tensors, so `numel` counts floats.
The reference implementation stores one complex tensor, so `numel` counts complex numbers --
twice the storage per element.  Matching the two by `numel` therefore hands the reference
twice the band, which is what the five-way table did.
"""
import sys

sys.path.insert(0, "/home/anirudh")
sys.path.insert(0, "/home/anirudh/fno/nop")

import torch

from neuralop.models import FNO as TheirFNO
from torch_fno.fno.model import FNO as OurFNO

CASES = [("Burgers", (17, 16), 24, 16, 22), ("Darcy-32", (32, 32), 24, 16, 22),
         ("Darcy-64", (64, 64), 24, 16, 22), ("NS-128", (128, 128), 64, 64, 78)]


def real_params(mod):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in mod.parameters())


def their_cells(m):
    """Their kept band: m rows about zero, and the real-input half of the columns."""
    return m * (m // 2 + 1)


print(f"{'case':10} {'our cells':>10} {'our real p':>12} | "
      f"{'m':>4} {'cells':>7} {'real p':>12} {'ratio':>7}")
print("-" * 78)

for name, shape, ch, modes, their_m in CASES:
    ours = OurFNO(1, 1, ch, modes, 4)
    our_cells = modes * (modes // 2)
    our_p = real_params(ours)

    for tag, m in (("as run", their_m), ):
        theirs = TheirFNO(n_modes=(m, m), in_channels=1, out_channels=1,
                          hidden_channels=ch, n_layers=4)
        tp = real_params(theirs)
        print(f"{name:10} {our_cells:10d} {our_p:12,} | {m:4d} {their_cells(m):7d} "
              f"{tp:12,} {tp / our_p:6.2f}x   <- {tag}")
        del theirs

    # The m whose band is closest to ours, which is the match that should have been used.
    best = min(range(4, 200), key=lambda m: abs(their_cells(m) - our_cells))
    theirs = TheirFNO(n_modes=(best, best), in_channels=1, out_channels=1,
                      hidden_channels=ch, n_layers=4)
    tp = real_params(theirs)
    print(f"{'':10} {'':10} {'':12} | {best:4d} {their_cells(best):7d} "
          f"{tp:12,} {tp / our_p:6.2f}x   <- band-matched")
    print()
    del ours, theirs
