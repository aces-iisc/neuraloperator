"""
One Fourier Neural Operator, standard Li-2021 architecture, run as fast as a v5e allows.

        x ──► add (x,y) grid ──► LIFT (1x1 MLP, in_ch→channels)
              │
              ▼   ┌──────────────── Fourier block (× n_layers) ────────────────┐
              x ──┤  spectral: low modes ─► learned complex mix ─► back        ┐
                  │  skip:     1x1 conv (local, per-pixel)                     ├─► + ─► GELU ─► x
                  └────────────────────────────────────────────────────────────┘
              │
              ▼
        PROJECT (1x1 MLP, channels→out_ch) ──► u (B, out_ch, H, W)

The spectral path mixes globally, the skip path locally; they run in PARALLEL, are summed, then
activated.  The whole spectral path is real/imag matmuls, so complex64 never appears.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import F
from .layers import add_grid, channel_mlp, conv1x1, gelu, init_fno, n_params  # noqa: F401
from .spectral import spectral_conv


def forward(P, x, n_modes, dtype=F):
    """P = the whole weight tree, x = (B, in_ch, H, W) real field -> (B, out_ch, H, W).

    `dtype` is the precision the spectral path is carried in. F is the f32 default; BF halves the
    activation bytes through the transform while accumulation stays f32."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for block in P["blocks"]:
        spectral = spectral_conv(block["filter"], x, n_modes, dtype)
        x = gelu(spectral + conv1x1(block["skip"], x))
    return channel_mlp(P["proj"], x)


def save_params(P, path):
    np.savez(path, *[np.asarray(a) for a in jax.tree_util.tree_leaves(P)])
    return path


def load_params(path, channels, n_modes, n_layers=4, in_ch=1, out_ch=1, seed=0):
    """Rebuild a tree written by save_params.  The architecture args must match what was saved."""
    saved = np.load(path)
    leaves = [jnp.asarray(saved[f"arr_{i}"]) for i in range(len(saved.files))]
    skeleton = init_fno(jax.random.PRNGKey(seed), in_ch, out_ch, channels, n_modes, n_layers)
    _, treedef = jax.tree_util.tree_flatten(skeleton)
    return jax.tree_util.tree_unflatten(treedef, leaves)
