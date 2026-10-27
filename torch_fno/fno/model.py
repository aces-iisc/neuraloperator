"""One Fourier Neural Operator, standard Li-2021 architecture.

        x -> add (x,y) grid -> LIFT (1x1 MLP, in_ch -> channels)
             |
             v   +------------- Fourier block (x n_layers) -------------+
             x --|  spectral: low modes -> learned complex mix -> back  |-> + -> GELU -> x
                 |  skip:     1x1 conv (local, per pixel)               |
                 +------------------------------------------------------+
             |
             v
        PROJECT (1x1 MLP, channels -> out_ch) -> u (B, out_ch, H, W)

The spectral path mixes globally, the skip path locally; they run in parallel, are summed, then
activated. The whole spectral path is real/imag matmuls, so no complex dtype appears.
"""
import torch
import torch.nn as nn

from ..fft.core import BF, F
from .layers import ChannelMLP, Conv1x1, add_grid, gelu
from .spectral import CPSpectralConv, SeparableSpectralConv, SpectralConv

FILTERS = {"dense": SpectralConv, "ffno": SeparableSpectralConv, "tfno": CPSpectralConv}


class FourierBlock(nn.Module):
    def __init__(self, channels, n_modes, filter="dense", dtype=F, **kw):
        super().__init__()
        self.spectral = FILTERS[filter](channels, channels, n_modes, dtype=dtype, **kw)
        self.skip = Conv1x1(channels, channels)

    def forward(self, x):
        return gelu(self.spectral(x) + self.skip(x))


class FNO(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, channels=32, n_modes=16, n_layers=4,
                 filter="dense", dtype=F, **kw):
        super().__init__()
        self.n_modes, self.channels, self.dtype = n_modes, channels, dtype
        self.lift = ChannelMLP(in_ch + 2, 2 * channels, channels)
        self.blocks = nn.ModuleList(
            FourierBlock(channels, n_modes, filter, dtype, **kw) for _ in range(n_layers))
        self.project = ChannelMLP(channels, 2 * channels, out_ch)

    def forward(self, x):
        x = self.lift(add_grid(x))
        for block in self.blocks:
            x = block(x)
        return self.project(x)

    def prebuild(self, dtype=None):
        """Build every blocked filter once, for inference."""
        for block in self.blocks:
            if hasattr(block.spectral, "prebuild"):
                block.spectral.prebuild(dtype)
        return self

    def set_dtype(self, dtype):
        """Switch the precision the transform is carried in; accumulation stays f32."""
        self.dtype = dtype
        for block in self.blocks:
            block.spectral.dtype = dtype
            block.spectral._fft = None
        return self

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def compiled(self, **kw):
        """torch.compile with the defaults that suit this layer.

        The graph is a long chain of einsums with static shapes, so it fuses well and the
        kernel-launch overhead a dynamic graph pays per call disappears. `mode` is left to the
        caller because reduce-overhead captures CUDA graphs, which fixes the batch.
        """
        return torch.compile(self, **kw)
