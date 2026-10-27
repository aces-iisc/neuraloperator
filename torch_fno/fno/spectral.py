"""The three spectral convolutions, all real/imag, all computing only the modes they keep."""
import math

import torch
import torch.nn as nn

import torch.cuda

from ..fft.core import ACC, BF, F, stacked_rows
from ..fft.transform import FFT, inverse_cols, inverse_rows
from .layers import axis_block, spectral_block

# einsum letters: n=sample i=in-channel o=out-channel h=row w=col
#                 a=row mode (the 2m band)  d=col mode (the low m)  r=rank


class _SpectralBase(nn.Module):
    """Shared transform: rows to both corners, cols to the low modes, and back.

    Two layouts, same arithmetic and bit-identical results.

    `sample` keeps the field in (n, c, a, d) and contracts a middle axis at every step, so each
    einsum permutes before it can call a GEMM.  It is what the TPU wants, where the layout is
    free and the stacked operand is what fills the array.

    `mode` transposes once on the way in and once on the way out and contracts the LEADING axis
    in between, which is a plain matmul with nothing to permute.  One turn survives, between
    the row and column passes, because a separable 2-D transform contracts two different axes.
    Worth 1.09-1.20x on a 5070 and nothing on a CPU, so it is chosen by device unless asked.
    """

    def __init__(self, n_modes, dtype=F, fft=None, layout=None):
        super().__init__()
        self.n_modes = n_modes
        self.dtype = dtype
        self._fft = fft
        self.layout = layout

    def transform(self):
        if self._fft is None or self._fft.dtype is not self.dtype:
            self._fft = FFT(dtype=self.dtype)
        return self._fft

    def _layout(self, x):
        return self.layout or ("mode" if x.is_cuda else "sample")

    def bands(self, x):
        """Which rows each axis keeps.  Cached on the module: they depend only on the grid,
        and rebuilding them per call costs a numpy concatenate and a few hundred Python int
        conversions per layer, which is real time against a kernel measured in microseconds.
        """
        key = (x.shape[-2], x.shape[-1])
        cached = getattr(self, "_bands", None)
        if cached is None or cached[0] != key:
            t = self.transform()
            row = t.band(self.n_modes, x.shape[-2])
            col = t.half_band(self.n_modes)
            self._bands = (key, row, col,
                           tuple(int(v) for v in row), tuple(int(v) for v in col))
        return self._bands[1], self._bands[2]

    @property
    def band_keys(self):
        return self._bands[3], self._bands[4]

    # -- sample-major ----------------------------------------------------------------
    def analyse(self, x):
        t = self.transform()
        row_band, col_band = self.bands(x)
        rows_re, rows_im = t.rows_forward(x, row_band)
        modes_re, modes_im = t.cols_forward(rows_re, rows_im, col_band, x.shape[-1])
        return torch.cat([modes_re.to(self.dtype), modes_im.to(self.dtype)], dim=1)

    def synthesise(self, mixed, n_ch, row_band, col_band, n_rows, n_cols):
        t = self.transform()
        cols_re, cols_im = t.cols_inverse(mixed[:, :n_ch], mixed[:, n_ch:], col_band, n_cols)
        field = t.rows_inverse(cols_re, cols_im, row_band, n_rows)
        return (field / (n_rows * n_cols)).to(F)

    # -- mode-major ------------------------------------------------------------------
    def analyse_mode_major(self, x):
        """(n, c, h, w) -> (a, d, n, 2c), stacked real and imaginary on the channel axis."""
        n, c, h, w = x.shape
        row_band, col_band = self.bands(x)
        a, d = len(row_band), len(col_band)
        fwd_rows = stacked_rows(row_band, h, -1, self.dtype, x.device)
        fwd_cols = stacked_rows(col_band, w, -1, self.dtype, x.device)

        both = (fwd_rows @ x.permute(2, 0, 1, 3).reshape(h, n * c * w).to(self.dtype)).to(ACC)
        rows_re = both[:a].reshape(a, n, c, w).permute(3, 0, 1, 2).reshape(w, a * n * c)
        rows_im = both[a:].reshape(a, n, c, w).permute(3, 0, 1, 2).reshape(w, a * n * c)
        from_re = (fwd_cols @ rows_re.to(self.dtype)).to(ACC)
        from_im = (fwd_cols @ rows_im.to(self.dtype)).to(ACC)
        modes_re = (from_re[:d] - from_im[d:]).reshape(d, a, n, c)
        modes_im = (from_re[d:] + from_im[:d]).reshape(d, a, n, c)
        return torch.cat([modes_re, modes_im], dim=3).permute(1, 0, 2, 3).to(self.dtype)

    def synthesise_mode_major(self, mixed, n_ch, x_shape):
        """(a, d, n, 2o) -> (n, o, h, w)."""
        n, _, h, w = x_shape
        row_band, col_band = self.bands_cached
        a, d = len(row_band), len(col_band)
        device = mixed.device
        row_key, col_key = self.band_keys
        stacked = torch.cat([mixed[..., :n_ch], mixed[..., n_ch:]], dim=1)      # (a, 2d, n, o)
        cols = inverse_cols(col_key, w, device) @ \
            stacked.permute(1, 0, 2, 3).reshape(2 * d, a * n * n_ch).to(F)
        both = torch.cat([cols[:w].reshape(w, a, n, n_ch),
                          cols[w:].reshape(w, a, n, n_ch)], dim=1)
        field = inverse_rows(row_key, h, device) @ \
            both.permute(1, 0, 2, 3).reshape(2 * a, w * n * n_ch).to(F)
        # .contiguous() is not cosmetic: the permute leaves the field with the channel axis
        # strided, and every downstream elementwise kernel -- the skip add, the GELU, the next
        # block's transform -- then reads it uncoalesced.  One copy here is cheaper than
        # paying for the bad layout in all of them.
        return (field.reshape(h, w, n, n_ch).permute(2, 3, 0, 1) / (h * w)).contiguous().to(F)

    def forward(self, x):
        self.bands_cached = self.bands(x)
        if self._layout(x) == "mode":
            mixed = self.mix(self.analyse_mode_major(x), "mode")
            return self.synthesise_mode_major(mixed, self.out_ch, x.shape)
        row_band, col_band = self.bands_cached
        return self.synthesise(self.mix(self.analyse(x), "sample"), self.out_ch,
                               row_band, col_band, x.shape[-2], x.shape[-1])


class SpectralConv(_SpectralBase):
    """Dense filter: one learned complex weight per (channel pair, mode pair)."""

    def __init__(self, in_ch, out_ch, n_modes, dtype=F, fft=None):
        super().__init__(n_modes, dtype, fft)
        std = math.sqrt(2.0 / (in_ch + out_ch))
        shape = (n_modes, n_modes // 2, in_ch, out_ch)
        self.w_re = nn.Parameter(std * torch.randn(shape))
        self.w_im = nn.Parameter(std * torch.randn(shape))
        self.out_ch = out_ch
        self.register_buffer("w_block", None, persistent=False)

    def prebuild(self, dtype=None):
        """Build the blocked filter once, for inference. Rebuilding it costs a materialisation
        of 8*C^2*m^2/2 bytes on every call."""
        with torch.no_grad():
            self.w_block = spectral_block(self.w_re, self.w_im, dtype or self.dtype)
        return self

    def mix(self, z, layout):
        block = self.w_block
        if block is None:
            block = spectral_block(self.w_re, self.w_im, self.dtype)
        elif block.dtype != self.dtype:
            block = block.to(self.dtype)
        # One real block matmul [[re, -im], [im, re]] instead of four products: same
        # arithmetic, one call, twice the contracted channels on the unit.
        spec = "adio,adni->adno" if layout == "mode" else "adio,niad->noad"
        return torch.einsum(spec, block, z).to(ACC)


class SeparableSpectralConv(_SpectralBase):
    """F-FNO: one weight per axis, added. W[a, d] = W_row[a] + W_col[d]."""

    def __init__(self, in_ch, out_ch, n_modes, dtype=F, fft=None):
        super().__init__(n_modes, dtype, fft)
        std = math.sqrt(2.0 / (in_ch + out_ch))
        kr, ki = (std * torch.randn(n_modes, in_ch, out_ch) for _ in range(2))
        lr, li = (std * torch.randn(n_modes // 2, in_ch, out_ch) for _ in range(2))
        self.k_block = nn.Parameter(axis_block(kr, ki))
        self.l_block = nn.Parameter(axis_block(lr, li))
        self.out_ch = out_ch

    def mix(self, z, layout):
        row, col = (("aio,adni->adno", "dio,adni->adno") if layout == "mode"
                    else ("aio,niad->noad", "dio,niad->noad"))
        return (torch.einsum(row, self.k_block.to(self.dtype), z)
                + torch.einsum(col, self.l_block.to(self.dtype), z)).to(ACC)


class CPSpectralConv(_SpectralBase):
    """TFNO: a rank-R CP filter in place of the dense one.

    The factors are the parameters, not the blocks the contraction uses: a blocked form has
    structural zeros, and training one directly would fill them in and stop it being a
    factorisation. A CP entry is a product of four factors, so drawing each at the dense scale
    leaves the reconstructed filter s^4 instead of s; for R rank-one terms Var(W) = R s^8, so
    the factors are drawn at (sigma^2 / R)^(1/8).
    """

    def __init__(self, in_ch, out_ch, n_modes, rank=8, dtype=F, fft=None):
        super().__init__(n_modes, dtype, fft)
        scale = ((2.0 / (in_ch + out_ch)) / rank) ** 0.125
        for part in ("re", "im"):
            self.register_parameter(f"{part}_A", nn.Parameter(scale * torch.randn(in_ch, rank)))
            self.register_parameter(f"{part}_B", nn.Parameter(scale * torch.randn(out_ch, rank)))
            self.register_parameter(f"{part}_C", nn.Parameter(scale * torch.randn(n_modes, rank)))
            self.register_parameter(f"{part}_D", nn.Parameter(
                scale * torch.randn(n_modes // 2, rank)))
        self.out_ch, self.rank = out_ch, rank

    def mix(self, z, layout):
        # Stacked so one contraction emits the whole 4R operand and the four output products
        # are a single (2*out, 4R) matmul, rather than four rank-R applications.
        a_stack = torch.cat([self.re_A.T, self.im_A.T], dim=0)
        zero = torch.zeros_like(a_stack)
        a_big = torch.cat([torch.cat([a_stack, zero], dim=1),
                           torch.cat([zero, a_stack], dim=1)], dim=0)
        cd_re = self.re_C.T[:, :, None] * self.re_D.T[:, None, :]
        cd_im = self.im_C.T[:, :, None] * self.im_D.T[:, None, :]
        cd = torch.cat([cd_re, cd_im, cd_re, cd_im], dim=0)                 # (4r, a, d)
        zeros_b = torch.zeros_like(self.re_B)
        b_block = torch.cat([
            torch.cat([self.re_B, zeros_b, zeros_b, -self.im_B], dim=1),
            torch.cat([zeros_b, self.im_B, self.re_B, zeros_b], dim=1)], dim=0)

        if layout == "mode":
            v = torch.einsum("ri,adni->adnr", a_big.to(self.dtype), z).to(ACC)
            v = v * cd.to(ACC).permute(1, 2, 0).unsqueeze(2)                # (a, d, 1, 4r)
            return torch.einsum("or,adnr->adno", b_block.to(self.dtype), v.to(self.dtype)
                                ).to(ACC)
        v = torch.einsum("ri,niad->nrad", a_big.to(self.dtype), z).to(ACC)
        v = v * cd.to(ACC).unsqueeze(0)
        return torch.einsum("or,nrad->noad", b_block.to(self.dtype), v.to(self.dtype)).to(ACC)
