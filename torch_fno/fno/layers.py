import math

import torch
import torch.nn as nn
import torch.nn.functional as f

from ..fft.core import ACC, F

def gelu(x):
    """The tanh approximation, which is what jax.nn.gelu uses by default.

    torch defaults to the exact erf form instead, and the two differ by about 1e-3 relative --
    enough to show up as a 2.6e-4 discrepancy against the reference after four blocks.
    """
    return f.gelu(x, approximate="tanh")


def add_grid(x):
    """Append normalised (x, y) coordinate channels. (B, C, H, W) -> (B, C+2, H, W)."""
    batch, _, n_rows, n_cols = x.shape
    rows = torch.linspace(0, 1, n_rows, device=x.device, dtype=x.dtype)
    cols = torch.linspace(0, 1, n_cols, device=x.device, dtype=x.dtype)
    grid = torch.stack([rows[:, None].expand(n_rows, n_cols),
                        cols[None, :].expand(n_rows, n_cols)])
    return torch.cat([x, grid.unsqueeze(0).expand(batch, 2, n_rows, n_cols)], dim=1)


class Conv1x1(nn.Module):
    """The same linear map at every pixel, issued as a 1x1 convolution.

    Mathematically an einsum over the channel axis, but conv2d reaches cuDNN's pointwise path
    and keeps whatever memory format the input arrived in.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        std = math.sqrt(2.0 / (in_ch + out_ch))
        self.weight = nn.Parameter(std * torch.randn(out_ch, in_ch))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        return f.conv2d(x, self.weight[:, :, None, None], self.bias)


class ChannelMLP(nn.Module):
    """conv1x1 -> GELU -> conv1x1, used to lift and to project."""

    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.l1, self.l2 = Conv1x1(in_ch, hidden_ch), Conv1x1(hidden_ch, out_ch)

    def forward(self, x):
        return self.l2(gelu(self.l1(x)))


def spectral_block(w_re, w_im, dtype=None):
    """The 2x2 real block form of the complex filter, [[re, -im], [im, re]].

    Channels are the two minor axes, so the natural layout is already the one the matmul
    contracts over and no relayout copy is emitted.
    """
    wr = w_re if dtype is None else w_re.to(dtype)
    wi = w_im if dtype is None else w_im.to(dtype)
    return torch.cat([torch.cat([wr, -wi], dim=2), torch.cat([wi, wr], dim=2)], dim=3)


def axis_block(w_re, w_im):
    """[[re, -im], [im, re]] for one axis weight, channels stacked and kept minor."""
    left = torch.cat([w_re, -w_im], dim=1)
    right = torch.cat([w_im, w_re], dim=1)
    return torch.cat([left, right], dim=2)


# -- losses, matching neuralop's reductions -------------------------------------------
def mse(pred, y):
    return torch.mean((pred - y) ** 2)


def lp_rel(pred, y, d=2):
    """LpLoss(d=2, p=2): relative L2 per sample, SUMMED over the batch."""
    flat = lambda a: a.reshape(*a.shape[:-d], -1)
    pf, yf = flat(pred), flat(y)
    num = torch.sqrt(torch.sum((pf - yf) ** 2, dim=-1))
    den = torch.sqrt(torch.sum(yf ** 2, dim=-1))
    return torch.sum(num / den)


def h1_rel(pred, y, d=2):
    """H1Loss(d=2): value and first derivatives, relative, summed over the batch.

    Derivatives are periodic central differences with spacing 1/N per axis.
    """
    flat = lambda a: a.reshape(*a.shape[:-d], -1)
    axes = [-2, -1]
    steps = [1.0 / pred.shape[a] for a in axes]

    def terms(a):
        out = [a]
        for ax, h in zip(axes, steps):
            out.append((torch.roll(a, -1, ax) - torch.roll(a, 1, ax)) / (2.0 * h))
        return out

    diff = norm = 0.0
    for tp, ty in zip(terms(pred), terms(y)):
        diff = diff + torch.sum((flat(tp) - flat(ty)) ** 2, dim=-1)
        norm = norm + torch.sum(flat(ty) ** 2, dim=-1)
    return torch.sum(torch.sqrt(diff) / torch.sqrt(norm))


LOSSES = {"mse": mse, "l2": lp_rel, "h1": h1_rel}
