"""The three channel filters with the layer optimisations applied: blocking, stacking, ADIO.

`factorized.py` holds the plain forms, written the way each paper states them.  These are the
same filters carrying the optimisations Section 5 applies to the dense one, so that a comparison
between variants is not really a comparison between an optimised filter and unoptimised ones.

The complex product is blocked: [[Wre, -Wim], [Wim, Wre]] against [Zre ; Zim] is one matmul
where the direct form is four, and the contracted channel axis is stacked to 2*C so the matrix
unit sees a wider operand.  Modes are the MAJOR axes and channels the two MINOR ones (ADIO), so
the natural row-major layout is already what the matmul contracts over.
"""
import jax.numpy as jnp

from ...fft.core import ACC

_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)


def _block(w_re, w_im, mode_axes):
    """[[re, -im], [im, re]] with the channel axes stacked, then moved minor (ADIO).

    `w_re` is (in, out, *modes); the result is (*modes, 2*in, 2*out).
    """
    left = jnp.concatenate([w_re, -w_im], axis=0)       # (2in, out, *modes)
    right = jnp.concatenate([w_im, w_re], axis=0)       # (2in, out, *modes)
    block = jnp.concatenate([left, right], axis=1)      # (2in, 2out, *modes)
    order = tuple(range(2, 2 + mode_axes)) + (0, 1)     # modes major, channels minor
    return jnp.transpose(block, order)


# ── dense ───────────────────────────────────────────────────────────────────────────
def dense_block(w):
    return {"block": _block(w["w_re"], w["w_im"], 2)}          # (k, l, 2i, 2o)


def dense_filter_opt(w, fr, fi):
    n_out = w["block"].shape[3] // 2
    z = jnp.concatenate([fr, fi], axis=1)                       # (b, 2i, k, l)
    out = _E("klio,bikl->bokl", w["block"], z)
    return out[:, :n_out], out[:, n_out:]


# ── F-FNO: one blocked weight per axis, summed ──────────────────────────────────────
def sep_block(w):
    return {"k": _block(w["kr"], w["ki"], 1),                   # (k, 2i, 2o)
            "l": _block(w["lr"], w["li"], 1)}                   # (l, 2i, 2o)


def sep_filter_opt(w, fr, fi):
    n_out = w["k"].shape[2] // 2
    z = jnp.concatenate([fr, fi], axis=1)
    out = _E("kio,bikl->bokl", w["k"], z) + _E("lio,bikl->bokl", w["l"], z)
    return out[:, :n_out], out[:, n_out:]


# ── TFNO: contracted axis minor on every factor ─────────────────────────────────────
def cp_block(w):
    """A contracts over the input channel, so store it with that axis minor."""
    out = {}
    for part in ("re_", "im_"):
        out[part + "A"] = w[part + "A"].T                       # (r, i): i minor
        out[part + "B"] = w[part + "B"]                         # (o, r): r already minor
        out[part + "C"] = w[part + "C"].T                       # (r, k)
        out[part + "D"] = w[part + "D"].T                       # (r, l)
    return out


def _cp_mix_opt(A_t, B, C_t, D_t, f):
    t = _E("ri,bikl->brkl", A_t, f)
    t = t * C_t[None, :, :, None] * D_t[None, :, None, :]
    return _E("or,brkl->bokl", B, t)


def cp_filter_opt(w, fr, fi):
    g = lambda p, f: _cp_mix_opt(w[p + "A"], w[p + "B"], w[p + "C"], w[p + "D"], f)
    return g("re_", fr) - g("im_", fi), g("im_", fr) + g("re_", fi)


# ── second round, driven by the optimised HLO ───────────────────────────────────────
# F-FNO: "kio,bikl->bokl" and "lio,bikl->bokl" contract the same channel axis but batch over
# different mode axes, so XLA transposes the activation for the second one -- 202 MB on
# Darcy-64, more than the dense filter moves in total.  The two weights only ever appear
# summed, and W[k,l] = Wk[k] + Wl[l] is an outer sum, so folding them into one weight collapses
# both einsums into the single contraction dense already uses and the transpose disappears.
# The folded weight is (k, l, 2i, 2o), so it is only worth forming while that fits.
SEP_FOLD_BYTES = 64 << 20


def sep_block_folded(w):
    k_blk = _block(w["kr"], w["ki"], 1)                 # (k, 2i, 2o)
    l_blk = _block(w["lr"], w["li"], 1)                 # (l, 2i, 2o)
    folded = k_blk[:, None, :, :] + l_blk[None, :, :, :]
    if folded.size * 4 > SEP_FOLD_BYTES:
        return {"k": k_blk, "l": l_blk}                 # too big to materialise; keep two
    return {"kl": folded}                               # (k, l, 2i, 2o)


def sep_filter_folded(w, fr, fi):
    z = jnp.concatenate([fr, fi], axis=1)
    if "kl" in w:
        out = _E("klio,bikl->bokl", w["kl"], z)
        n_out = w["kl"].shape[3] // 2
    else:
        out = _E("kio,bikl->bokl", w["k"], z) + _E("lio,bikl->bokl", w["l"], z)
        n_out = w["k"].shape[2] // 2
    return out[:, :n_out], out[:, n_out:]


# TFNO: cp_filter_opt calls the rank mix FOUR times -- (re,fr), (im,fi), (im,fr), (re,fi) --
# so the first contraction runs four times over the same two activations.  Stacking the rank
# factor over the real/imaginary parts and the activation over its two halves turns those four
# matmuls into one, with the rank axis carrying which combination each slice is.
def cp_block_stacked(w):
    out = cp_block(w)
    out["A_stack"] = jnp.concatenate([out["re_A"], out["im_A"]], axis=0)   # (2r, i)
    return out


def cp_filter_stacked(w, fr, fi):
    rank = w["re_A"].shape[0]
    z = jnp.concatenate([fr, fi], axis=0)                       # (2b, i, k, l)
    t = _E("ri,bikl->brkl", w["A_stack"], z)                    # (2b, 2r, k, l)
    batch = fr.shape[0]

    def part(p, half):
        """The (p-factor, half-of-z) slice, with C and D applied."""
        rows = slice(0, rank) if p == "re_" else slice(rank, 2 * rank)
        block = t[half * batch:(half + 1) * batch, rows]
        block = block * w[p + "C"][None, :, :, None] * w[p + "D"][None, :, None, :]
        return _E("or,brkl->bokl", w[p + "B"], block)

    return part("re_", 0) - part("im_", 1), part("im_", 0) + part("re_", 1)


# ── third round: block the CP product the way the dense one is blocked ──────────────
# cp_filter_opt applies the rank-R operator four times (Wre·fre, Wim·fim, Wim·fre, Wre·fim),
# which is eight einsums and eight full-size temporaries -- the same four-product form the dense
# filter stopped using in Section 5.2.  Two things fix it.
#
#   1. Both halves of the input contract against the SAME stacked A, so stacking [fre ; fim] on
#      the batch axis turns four Cin contractions into one.
#   2. The four rank-space results are then a single 4R-wide operand, and the four B products
#      that follow are one (2Cout x 4R) matmul.  On a 128-wide unit a 4R = 32 contraction beats
#      four separate R = 8 contractions even though the block is half zeros.
#
# Net: two einsums instead of eight, contracting 32 and producing 48 rather than contracting 8.
def cp_block_fused(w):
    """Stack A over the rank axis and prebuild the (2Cout x 4R) output block and the mode scale.

    C[k,r]·D[l,r] does not depend on the data, so it is formed once here rather than as two
    broadcast multiplies on every call."""
    rank = w["re_A"].shape[1]
    cout = w["re_B"].shape[0]
    out = {
        # (2r, i): one contraction for both halves of the input
        "A_stack": jnp.concatenate([w["re_A"].T, w["im_A"].T], axis=0),
    }
    # the mode scale for the 4R operand, in the order [a, b, c, d] below
    cd_re = w["re_C"].T[:, :, None] * w["re_D"].T[:, None, :]      # (r, k, l)
    cd_im = w["im_C"].T[:, :, None] * w["im_D"].T[:, None, :]
    out["cd"] = jnp.concatenate([cd_re, cd_im, cd_re, cd_im], axis=0)   # (4r, k, l)

    # [[Br, 0, 0, -Bi], [0, Bi, Br, 0]] against [a, b, c, d]
    z = jnp.zeros_like(w["re_B"])
    g_re = jnp.concatenate([w["re_B"], z, z, -w["im_B"]], axis=1)       # (o, 4r)
    g_im = jnp.concatenate([z, w["im_B"], w["re_B"], z], axis=1)
    out["B_block"] = jnp.concatenate([g_re, g_im], axis=0)              # (2o, 4r)
    out["rank"], out["cout"] = rank, cout
    return out


def cp_filter_fused(w, fr, fi):
    """Two einsums: contract the input channel once, then the rank axis once."""
    batch = fr.shape[0]
    z = jnp.concatenate([fr, fi], axis=0)                      # (2b, i, k, l)
    t = _E("ri,bikl->brkl", w["A_stack"], z)                   # (2b, 2r, k, l)
    # [a, b] from the real half, [c, d] from the imaginary half
    v = jnp.concatenate([t[:batch], t[batch:]], axis=1)        # (b, 4r, k, l)
    v = v * w["cd"][None]
    mixed = _E("or,brkl->bokl", w["B_block"], v)               # (b, 2o, k, l)
    cout = w["cout"]
    return mixed[:, :cout], mixed[:, cout:]


# ── fourth round, driven by the fused trace ────────────────────────────────────────
# cp_filter_fused cut MXU time by 2.7x on Darcy and 3.4x on NS, and then spent half of what it
# saved on copies: stacking the input on the BATCH axis means the two halves come back separated
# by batch, so rebuilding the 4R operand is a slice and a concat on a full-size array.
#
# Stacking on the CHANNEL axis instead removes that.  With z = [fre ; fim] along channels and a
# block-diagonal A, one contraction emits the 4R operand already in the right order, and the
# contracted axis doubles from Cin to 2*Cin as a side effect -- the same widening the dense
# filter gets from blocking.
def cp_block_fused2(w):
    rank, cin = w["re_A"].shape[1], w["re_A"].shape[0]
    out = cp_block_fused(w)                       # reuse cd, B_block, rank, cout
    a_stack = out["A_stack"]                      # (2r, i)
    zero = jnp.zeros_like(a_stack)
    # [[A_stack, 0], [0, A_stack]] : (4r, 2i), so one contraction gives [a, b, c, d]
    out["A_big"] = jnp.concatenate(
        [jnp.concatenate([a_stack, zero], axis=1),
         jnp.concatenate([zero, a_stack], axis=1)], axis=0)
    return out


def cp_filter_fused2(w, fr, fi):
    """Two einsums and one concat: the 4R operand comes straight out of the first contraction."""
    z = jnp.concatenate([fr, fi], axis=1)                      # (b, 2i, k, l)
    v = _E("ri,bikl->brkl", w["A_big"], z)                     # (b, 4r, k, l), already ordered
    v = v * w["cd"][None]
    mixed = _E("or,brkl->bokl", w["B_block"], v)               # (b, 2o, k, l)
    cout = w["cout"]
    return mixed[:, :cout], mixed[:, cout:]
