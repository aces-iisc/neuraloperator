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
    shape = (n_modes, n_modes // 2, in_ch, out_ch)
    return {"w_re": std * jax.random.normal(k_re, shape),
            "w_im": std * jax.random.normal(k_im, shape)}


def spectral_block(w_re, w_im, dtype=None):
    """The 2x2 real block form of the complex filter, [[re, -im], [im, re]].

    Channels are the two MINOR axes, so the natural row-major layout is already the one the
    matmul contracts over and no relayout copy is emitted."""
    wr = w_re if dtype is None else w_re.astype(dtype)
    wi = w_im if dtype is None else w_im.astype(dtype)
    return jnp.concatenate([jnp.concatenate([wr, -wi], axis=2),
                            jnp.concatenate([wi, wr], axis=2)], axis=3)


def prebuild_blocks(P, dtype=None):
    """Return a copy of the weight tree with each filter's block matrix built once.

    Passing the block in as an argument keeps it off the per-call path; rebuilding it inside
    the kernel materialises 8*C^2*NM^2 bytes on every call."""
    out = dict(P)
    out["blocks"] = [dict(b, filter=dict(b["filter"],
                     w_block=spectral_block(b["filter"]["w_re"], b["filter"]["w_im"], dtype)))
                     for b in P["blocks"]]
    return out


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


def lp_rel(pred, y, d=2):
    """neuralop's LpLoss(d=2, p=2): relative L2 per sample, summed over the batch."""
    flat = lambda a: a.reshape(a.shape[:-d] + (-1,))
    pf, yf = flat(pred), flat(y)
    num = jnp.sqrt(jnp.sum((pf - yf) ** 2, axis=-1))
    den = jnp.sqrt(jnp.sum(yf ** 2, axis=-1))
    return jnp.sum(num / den)


def h1_rel(pred, y, d=2):
    """neuralop's H1Loss(d=2): value and first derivatives, relative, summed over the batch.

    Derivatives are periodic central differences with spacing 1/N per axis, and the reduction
    is a sum over the batch rather than a mean -- both verified against the torch implementation."""
    flat = lambda a: a.reshape(a.shape[:-d] + (-1,))
    axes = [-2, -1]
    steps = [1.0 / pred.shape[a] for a in axes]

    def terms(a):
        out = [a]
        for ax, h in zip(axes, steps):
            out.append((jnp.roll(a, -1, ax) - jnp.roll(a, 1, ax)) / (2.0 * h))
        return out

    diff = 0.0
    norm = 0.0
    for tp, ty in zip(terms(pred), terms(y)):
        diff = diff + jnp.sum((flat(tp) - flat(ty)) ** 2, axis=-1)
        norm = norm + jnp.sum(flat(ty) ** 2, axis=-1)
    return jnp.sum(jnp.sqrt(diff) / jnp.sqrt(norm))


LOSSES = {"mse": mse, "l2": lp_rel, "h1": h1_rel}


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
