"""
The plain-JAX BASELINE FNO2d.  The spectral conv is the textbook version:
  jnp.fft.rfft2  ->  keep the low ±m band (2 corners)  ->  complex channel mix (jnp.einsum / gemm)
  ->  jnp.fft.irfft2.
Uses complex64 throughout — this is the straightforward reference we check `ours` against.
"""
import jax.numpy as jnp

from ...fno.layers import add_grid, channel_mlp, conv1x1, gelu, init_fno


def spectral_conv(w, x, n_modes):
    """2-corner spectral conv via jnp.fft (complex64).  x (B,C,H,W) real -> (B,Cout,H,W) real.
    Keeps k1 ∈ [0,m)∪[H-m,H) on rows (H), k2 ∈ [0,m) on cols (W) — the standard low-mode band."""
    half = n_modes // 2        # n_modes counts both corners
    B, C, H, W = x.shape
    x_ft = jnp.fft.rfft2(x, axes=(-2, -1))                      # (B, C, H, W//2+1) complex64
    weight = w["w_re"] + 1j * w["w_im"]                          # (2m, m, Cin, Cout) complex   # a = row mode, d = col mode
    out = jnp.zeros((B, w["w_re"].shape[3], H, W // 2 + 1), x_ft.dtype)

    mix = lambda Wm, Xm: jnp.einsum("adio,niad->noad", Wm, Xm)   # complex gemm, contract Cin -> Cout
    top = mix(weight[:half], x_ft[:, :, :half, :half])                    # +corner rows [0:m]
    bot = mix(weight[half:], x_ft[:, :, H - half:, :half])                # -corner rows [H-m:]
    out = out.at[:, :, :half, :half].set(top).at[:, :, H - half:, :half].set(bot)

    return jnp.fft.irfft2(out, s=(H, W), axes=(-2, -1))          # back to a real field


def forward(P, x, n_modes):
    """add (x,y) grid -> LIFT -> [spectral + skip -> GELU] * L -> PROJECT."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        x = gelu(spectral_conv(blk["filter"], x, n_modes) + conv1x1(blk["skip"], x))
    return channel_mlp(P["proj"], x)


init = init_fno   # same weight tree as ours (directly comparable)
