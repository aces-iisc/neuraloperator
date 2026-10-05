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


def forward(P, x, n_modes, dtype=F, conv=spectral_conv):
    """P = the whole weight tree, x = (B, in_ch, H, W) real field -> (B, out_ch, H, W).

    `dtype` is the precision the spectral path is carried in. F is the f32 default; BF halves the
    activation bytes through the transform while accumulation stays f32.
    `conv` swaps the spectral convolution for a variant with the rest of the graph held fixed,
    which is how one optimisation at a time gets isolated; it defaults to the shipped one."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for block in P["blocks"]:
        spectral = conv(block["filter"], x, n_modes, dtype)
        x = gelu(spectral + conv1x1(block["skip"], x))
    return channel_mlp(P["proj"], x)


def save_params(P, path):
    np.savez(path, *[np.asarray(a) for a in jax.tree_util.tree_leaves(P)])
    return path


def load_params(path, channels, n_modes, n_layers=4, in_ch=1, out_ch=1, seed=0, init=None):
    """Rebuild a tree written by save_params.  The architecture args must match what was saved.

    `init` builds the skeleton the leaves are poured back into, and has to be the same one that
    created them: a variant filter carries its own structure, and F-FNO's happens to have the
    same number of leaves as the dense one, so reading it back through the dense skeleton
    produces a tree that is wrong rather than one that fails to load.
    """
    saved = np.load(path)
    leaves = [jnp.asarray(saved[f"arr_{i}"]) for i in range(len(saved.files))]
    build = init or init_fno
    skeleton = build(jax.random.PRNGKey(seed), in_ch, out_ch, channels, n_modes, n_layers)
    want, treedef = jax.tree_util.tree_flatten(skeleton)
    if len(leaves) != len(want):
        raise ValueError(f"{path}: holds {len(leaves)} arrays, this architecture wants "
                         f"{len(want)} -- wrong channels/n_modes/n_layers, or wrong init")

    # Checkpoints written before the filter moved its channels to the minor axes hold
    # (in, out, modes, modes/2); bring them forward rather than refusing to load.
    fixed = []
    for i, (leaf, ref) in enumerate(zip(leaves, want)):
        if leaf.shape != ref.shape and leaf.ndim == 4 and leaf.transpose(2, 3, 0, 1).shape == ref.shape:
            leaf = leaf.transpose(2, 3, 0, 1)
        if leaf.shape != ref.shape:
            raise ValueError(f"{path}: array {i} is {leaf.shape}, skeleton wants {ref.shape}")
        fixed.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, fixed)
