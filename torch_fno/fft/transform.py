"""The real transform as one object, configured by the accelerator it runs on.

Output frequencies are independent, so evaluating a subset is exact rather than approximate:
`rows` chooses what gets built and nothing outside it is computed. A staged radix transform
cannot do that, which is why the two families are different methods, not two settings of one.
"""
import functools

import numpy as np
import torch

from ..arch import detect
from .core import ACC, F, rows_matrix, stacked_rows


@functools.lru_cache(maxsize=256)
def inverse_cols(cols, length, device):
    """The folded [[cos, -sin], [sin, cos]] block for the column inverse, built once.

    It depends only on the band and the length, so rebuilding it per call pays a cat and a
    transpose every time -- and the transposed view then forces an unaligned GEMM kernel
    (cutlass ..._nt_align1, on CUDA cores) instead of the aligned one.
    """
    fold = torch.where(torch.tensor(cols, device=device) == 0, 1.0, 2.0).to(F)
    cos, sin = rows_matrix(cols, length, +1, F, device)
    cos, sin = (cos * fold[:, None]).T, (sin * fold[:, None]).T
    return torch.cat([torch.cat([cos, -sin], dim=1),
                      torch.cat([sin, cos], dim=1)], dim=0).contiguous()


@functools.lru_cache(maxsize=256)
def inverse_rows(rows, length, device):
    """[cos, -sin] transposed for the row inverse, materialised contiguous once."""
    cos, sin = rows_matrix(rows, length, +1, F, device)
    return torch.cat([cos, -sin], dim=0).T.contiguous()


class FFT:
    def __init__(self, accel=None, dtype=F):
        self.accel = accel or detect()
        self.dtype = dtype

    # -- which rows ------------------------------------------------------------------
    @staticmethod
    def band(n_modes, length):
        """Both low-frequency corners: this axis is not conjugate-symmetric."""
        half = n_modes // 2
        return np.concatenate([np.arange(half), np.arange(length - half, length)])

    @staticmethod
    def half_band(n_modes):
        """One side only: the input is real, so the negative half is the conjugate mirror."""
        return np.arange(n_modes // 2)

    # -- the passes ------------------------------------------------------------------
    def rows_forward(self, x, rows):
        """Transform axis -2 at `rows`. (n, c, h, w) -> (re, im), each (n, c, |rows|, w)."""
        matrix = stacked_rows(rows, x.shape[-2], -1, self.dtype, x.device)
        both = torch.einsum("ah,nchw->ncaw", matrix, x.to(self.dtype)).to(ACC)
        keep = len(rows)
        return both[:, :, :keep], both[:, :, keep:]

    def cols_forward(self, re, im, cols, length):
        """Transform axis -1 at `cols` on a complex input, giving the kept mode block."""
        matrix = stacked_rows(cols, length, -1, self.dtype, re.device)
        over = lambda z: torch.einsum("dw,ncaw->ncad", matrix, z.to(self.dtype)).to(ACC)
        from_re, from_im = over(re), over(im)
        keep = len(cols)
        return (from_re[..., :keep] - from_im[..., keep:],
                from_re[..., keep:] + from_im[..., :keep])

    def cols_inverse(self, re, im, cols, length):
        """Columns back, folding in the Hermitian partners the one-sided axis left out."""
        # The inverse matrices stay f32: the data is f32 by this point -- the channel mix
        # accumulates there -- so narrowing them would cost accuracy and save no traffic.
        block = inverse_cols(tuple(int(c) for c in cols), length, re.device)
        out = torch.einsum("wd,noad->noaw", block, torch.cat([re, im], dim=-1).to(F)).to(ACC)
        return out[..., :length], out[..., length:]

    def rows_inverse(self, re, im, rows, length):
        """Rows back. The field is real, so only the real part survives and the two halves
        fold into a single matmul rather than two."""
        block = inverse_rows(tuple(int(r) for r in rows), length, re.device)
        return torch.einsum("ha,noaw->nohw", block,
                            torch.cat([re, im], dim=2).to(F)).to(ACC)

    # -- staging, for a transform that keeps the whole spectrum -----------------------
    def stages(self, length):
        radix, out = self.accel.radix, []
        while length > radix and length % radix == 0:
            out.append(radix)
            length //= radix
        return out, length

    def __repr__(self):
        return f"<FFT radix={self.accel.radix} dtype={self.dtype}>"
