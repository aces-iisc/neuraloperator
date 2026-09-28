import jax
import jax.numpy as jnp

from ..fft.core import F  # noqa: F401  (sets global matmul precision on import)

gelu = jax.nn.gelu


def add_grid(x):
    """Append normalized (x, y) coordinate channels so the net knows absolute position.
    (B, C, H, W) -> (B, C+2, H, W)."""
    batch, _, n_rows, n_cols = x.shape
    rows = jnp.broadcast_to(jnp.linspace(0, 1, n_rows)[:, None], (n_rows, n_cols))
    cols = jnp.broadcast_to(jnp.linspace(0, 1, n_cols)[None, :], (n_rows, n_cols))
    grid = jnp.broadcast_to(jnp.stack([rows, cols])[None], (batch, 2, n_rows, n_cols))
    return jnp.concatenate([x, grid], axis=1)


def conv1x1(w, x):
    """1x1 conv — the same linear map at every pixel.  w = {W (out, in), b (out,)}."""
    return jnp.einsum("oi,bihw->bohw", w["W"], x) + w["b"][None, :, None, None]


def channel_mlp(w, x):
    """2-layer pointwise MLP (conv1x1 -> GELU -> conv1x1), used to lift and to project."""
    return conv1x1(w["l2"], gelu(conv1x1(w["l1"], x)))


def _init_conv1x1(key, in_ch, out_ch):
    std = (2.0 / (in_ch + out_ch)) ** 0.5
    return {"W": std * jax.random.normal(key, (out_ch, in_ch)), "b": jnp.zeros((out_ch,))}


def _init_mlp(key, in_ch, hidden_ch, out_ch):
    k1, k2 = jax.random.split(key)
    return {"l1": _init_conv1x1(k1, in_ch, hidden_ch), "l2": _init_conv1x1(k2, hidden_ch, out_ch)}


def _init_spectral_weight(key, in_ch, out_ch, n_modes):
    """The learned complex filter, as (w_re, w_im) of shape (in, out, n_modes, n_modes//2).

    `n_modes` is the TOTAL number of wavenumbers kept on an axis, matching the convention in
    neuralop's SpectralConv: half of them negative and half non-negative. The rows therefore span
    the full n_modes band, while the columns keep only n_modes//2 because that axis is rfft'd and
    its negative half is redundant."""
    std = (2.0 / (in_ch + out_ch)) ** 0.5
    k_re, k_im = jax.random.split(key)
    shape = (in_ch, out_ch, n_modes, n_modes // 2)
    return {"w_re": std * jax.random.normal(k_re, shape),
            "w_im": std * jax.random.normal(k_im, shape)}


def init_fno(key, in_ch=1, out_ch=1, channels=32, n_modes=16, n_layers=4):
    """The whole weight tree, laid out exactly as model.forward walks it."""
    keys = jax.random.split(key, n_layers + 2)
    block = lambda k: {"filter": _init_spectral_weight(jax.random.fold_in(k, 0), channels, channels, n_modes),
                       "skip": _init_conv1x1(jax.random.fold_in(k, 1), channels, channels)}
    return {"lift": _init_mlp(keys[0], in_ch + 2, 2 * channels, channels),
            "blocks": [block(keys[i + 1]) for i in range(n_layers)],
            "proj": _init_mlp(keys[-1], channels, 2 * channels, out_ch)}


def n_params(P):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(P)))


def mse(pred, y):
    return jnp.mean((pred - y) ** 2)


def rel_l2(pred, y):
    """Mean relative L2 error over the batch — the metric the FNO papers report."""
    err = jnp.sqrt(jnp.sum((pred - y) ** 2, axis=(-1, -2, -3)))
    norm = jnp.sqrt(jnp.sum(y ** 2, axis=(-1, -2, -3)))
    return jnp.mean(err / norm)


def adam_init(P):
    zeros_like_tree = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
    return {"m": zeros_like_tree(P), "v": zeros_like_tree(P), "t": 0}


def adam_step(P, grads, state, lr=1e-3, weight_decay=1e-4, b1=0.9, b2=0.999, eps=1e-8):
    """AdamW: decoupled weight decay, applied to the parameters rather than the gradient."""
    t = state["t"] + 1
    tree_map = jax.tree_util.tree_map
    m = tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    m_hat = tree_map(lambda m: m / (1 - b1 ** t), m)
    v_hat = tree_map(lambda v: v / (1 - b2 ** t), v)
    P = tree_map(lambda p, mh, vh: p - lr * (mh / (jnp.sqrt(vh) + eps) + weight_decay * p),
                 P, m_hat, v_hat)
    return P, {"m": m, "v": v, "t": t}
