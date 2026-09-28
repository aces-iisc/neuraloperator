"""
A staged real FFT that picks its own factorisation instead of always splitting to the smallest
leaf.

The sweep factors N = RADIX^L * m, and the leaf is an m x m DFT that the MXU pads to a
RADIX x RADIX tile. Always splitting as far as possible (what fft.core.rfft_staged does) can leave
m at 2 or 4, which wastes almost the whole tile and applies it to an enormous batch. Sometimes the
cheapest plan is not to stage at all: one direct N x N DFT has more arithmetic but packs its tiles
perfectly and touches the data once.

`rfft_auto` costs every reachable plan in padded tiles and runs the cheapest.
"""
import jax.numpy as jnp

from ...fft.core import ACC, F, RADIX, complex_matmul, dft_matrix_reim, leaf_matmul, stage_matmul

_ceil = lambda a, b: -(-a // b)


def plans(n):
    """Every reachable leaf m = n / RADIX^L, largest (fewest stages) first."""
    out, m, L = [], n, 0
    while True:
        out.append((L, m))
        if m % RADIX or m // RADIX < 1:
            break
        m //= RADIX
        L += 1
    return out


WEIGHT_LOAD = 8   # column-tiles' worth of time to push one weight tile into the systolic array


def _matmul_cost(rows, contract, cols):
    """Cost of one matmul in column-tile units.

    The MXU loads a weight tile into the array and then streams columns through it. Counting only
    padded tile-ops gets this wrong, because a matmul with a big matrix and few columns pays the
    weight load over and over with almost nothing to amortize it against. A direct N x N DFT on
    only K columns is exactly that shape: at N = 1024, K = 256 it has 64 weight tiles and 2 column
    tiles, and measures 10x worse per tile than the staged plan. WEIGHT_LOAD is that fixed cost,
    calibrated against the sweep in results/.
    """
    weight_tiles = _ceil(rows, RADIX) * _ceil(contract, RADIX)
    return weight_tiles * (_ceil(cols, RADIX) + WEIGHT_LOAD)


def tile_cost(n, k, L, m):
    """Cost of the plan (L stages then an m-leaf) on a real input, in column-tile units."""
    total, batch, n_cols = 0, k, n // RADIX
    for i in range(L):
        cnt = 2 if i == 0 else 3           # the first stage sees a real input
        total += cnt * _matmul_cost(RADIX, RADIX, n_cols * batch)
        batch *= RADIX
        n_cols = max(1, n_cols // RADIX)
    leaf_cnt = 2 if L == 0 else 3          # a direct plan keeps the real-input saving at the leaf
    total += leaf_cnt * _matmul_cost(m, m, (n // m) * k)
    return total


def choose(n, k):
    """The plan with the fewest padded tiles."""
    return min(plans(n), key=lambda p: tile_cost(n, k, *p))


def rfft_plan(signal, L, m, half=True, dtype=F):
    """Run one specific plan: L radix stages, then an m-point leaf."""
    n, k = signal.shape
    cur_re, cur_im = signal.astype(dtype), None
    length, batch, stages, first = n, k, [], True

    for _ in range(L):
        n_cols = length // RADIX
        cos, sin, cps = dft_matrix_reim(RADIX, dtype)
        if first:
            blk = cur_re.reshape(RADIX, n_cols, batch)
            s_re, s_im = stage_matmul(cos, blk), stage_matmul(sin, blk)
            first = False
        else:
            s_re, s_im = complex_matmul(stage_matmul, cos, sin, cps,
                                        cur_re.reshape(RADIX, n_cols, batch),
                                        cur_im.reshape(RADIX, n_cols, batch))
        rows, cols = jnp.arange(RADIX)[:, None], jnp.arange(n_cols)[None, :]
        ang = (-2 * jnp.pi * rows * cols / length).astype(F)
        tw_c, tw_s = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
        cur_re = (s_re * tw_c - s_im * tw_s).transpose(1, 0, 2).reshape(n_cols, RADIX * batch).astype(dtype)
        cur_im = (s_re * tw_s + s_im * tw_c).transpose(1, 0, 2).reshape(n_cols, RADIX * batch).astype(dtype)
        stages.append(RADIX)
        length, batch = n_cols, RADIX * batch

    cos, sin, cps = dft_matrix_reim(length, dtype)
    if first:                               # never staged: the input is still real, so 2 matmuls
        out_re, out_im = leaf_matmul(cos, cur_re), leaf_matmul(sin, cur_re)
    else:
        out_re, out_im = complex_matmul(leaf_matmul, cos, sin, cps, cur_re, cur_im)

    ln = length
    for r in reversed(stages):
        out_re = out_re.reshape(ln, r, -1).reshape(ln * r, -1)
        out_im = out_im.reshape(ln, r, -1).reshape(ln * r, -1)
        ln *= r
    if half:
        out_re, out_im = out_re[:n // 2], out_im[:n // 2]
    return out_re.astype(dtype), out_im.astype(dtype)


def rfft_auto(signal, half=True, dtype=F):
    """Pick the cheapest plan by tile count, then run it."""
    n, k = signal.shape
    L, m = choose(n, k)
    return rfft_plan(signal, L, m, half=half, dtype=dtype)


def rfft_leaf_first(signal, half=True, dtype=F):
    """The other factor order: do the SMALL factor first, leaving a full RADIX leaf.

    For N = r * RADIX this runs the length-r DFT first and finishes with a full RADIX x RADIX
    leaf, which is the kappa = 1 plan Section~\\ref{sec:measured} recommends. The awkward small
    factor does not disappear, it only moves: the r x r matrix is now applied to the wide batch in
    the stage instead of the leaf.
    """
    n, k = signal.shape
    r = n // RADIX
    if r < 1 or r * RADIX != n:
        raise ValueError(f"N={n} is not r*{RADIX}")

    # stage: length-r DFT along axis 0 of (r, RADIX, k), real input -> 2 matmuls
    cos, sin, _ = dft_matrix_reim(r, dtype)
    blk = signal.astype(dtype).reshape(r, RADIX, k)
    s_re, s_im = stage_matmul(cos, blk), stage_matmul(sin, blk)

    rows, cols = jnp.arange(r)[:, None], jnp.arange(RADIX)[None, :]
    ang = (-2 * jnp.pi * rows * cols / n).astype(F)
    tw_c, tw_s = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
    cur_re = (s_re * tw_c - s_im * tw_s).transpose(1, 0, 2).reshape(RADIX, r * k).astype(dtype)
    cur_im = (s_re * tw_s + s_im * tw_c).transpose(1, 0, 2).reshape(RADIX, r * k).astype(dtype)

    # leaf: full RADIX x RADIX DFT, complex now -> 3 matmuls
    cos, sin, cps = dft_matrix_reim(RADIX, dtype)
    out_re, out_im = complex_matmul(leaf_matmul, cos, sin, cps, cur_re, cur_im)
    out_re = out_re.reshape(RADIX, r, -1).reshape(RADIX * r, -1)
    out_im = out_im.reshape(RADIX, r, -1).reshape(RADIX * r, -1)
    if half:
        out_re, out_im = out_re[:n // 2], out_im[:n // 2]
    return out_re.astype(dtype), out_im.astype(dtype)


def rfft_two_factor(signal, leaf, half=True, dtype=F):
    """One stage of radix r = N/leaf, then a leaf-point DFT. Lets any leaf be tested at any N,
    which radix-RADIX-only stages cannot: they can only reach leaf = N / RADIX^L."""
    n, k = signal.shape
    r = n // leaf
    if r * leaf != n:
        raise ValueError(f"leaf {leaf} does not divide N={n}")
    if r == 1:                                   # no stage: one direct N x N DFT
        cos, sin, _ = dft_matrix_reim(n, dtype)
        out_re, out_im = leaf_matmul(cos, signal.astype(dtype)), leaf_matmul(sin, signal.astype(dtype))
        return (out_re[:n // 2], out_im[:n // 2]) if half else (out_re, out_im)

    cos, sin, _ = dft_matrix_reim(r, dtype)      # stage: length-r DFT, real input -> 2 matmuls
    blk = signal.astype(dtype).reshape(r, leaf, k)
    s_re, s_im = stage_matmul(cos, blk), stage_matmul(sin, blk)

    rows, cols = jnp.arange(r)[:, None], jnp.arange(leaf)[None, :]
    ang = (-2 * jnp.pi * rows * cols / n).astype(F)
    tw_c, tw_s = jnp.cos(ang)[:, :, None], jnp.sin(ang)[:, :, None]
    cur_re = (s_re * tw_c - s_im * tw_s).transpose(1, 0, 2).reshape(leaf, r * k).astype(dtype)
    cur_im = (s_re * tw_s + s_im * tw_c).transpose(1, 0, 2).reshape(leaf, r * k).astype(dtype)

    cos, sin, cps = dft_matrix_reim(leaf, dtype)  # leaf: complex now -> 3 matmuls
    out_re, out_im = complex_matmul(leaf_matmul, cos, sin, cps, cur_re, cur_im)
    out_re = out_re.reshape(leaf, r, -1).reshape(leaf * r, -1)
    out_im = out_im.reshape(leaf, r, -1).reshape(leaf * r, -1)
    return (out_re[:n // 2], out_im[:n // 2]) if half else (out_re, out_im)
