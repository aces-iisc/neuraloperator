import jax
import jax.numpy as jnp

# einsum index letters, used with these meanings everywhere in this package:
#   n  sample          i  in-channel      o  out-channel    c  channel (in == out)
#   h  grid row        w  grid col        a  row mode (the 2m band)   d  col mode (the low m)
#   r  CP rank
# The staged transform works on different tensors, so it uses its own letters rather than
# borrowing these with new meanings:
#   p  stage output    q  stage input (the radix)   s  stage column   t  packed batch
#   g  block group (the row reversal)

from ..arch import TPU_V5E

# Memory-bound on v5e: the extra MXU passes of true-f32 matmuls hide behind the HBM wait, so
# 'highest' precision is FREE here (measured Δtime 0, accuracy Δ +0.0008). Set it globally, once,
# for every module that imports this one (FFT + FNO + Pallas). No per-call precision= needed.
jax.config.update("jax_default_matmul_precision", "highest")

# The radix of a staged sweep and the width at which folding a stage transpose into the matmul
# starts to pay are both properties of the matrix unit, not of the transform. They default to
# the TPU v5e here so existing callers are unchanged; pass an Accelerator to retarget.
DEFAULT_ACCEL = TPU_V5E
RADIX = DEFAULT_ACCEL.radix          # one radix-RADIX stage per sweep = the unit's tile edge
F = jnp.float32
BF = jnp.bfloat16
I8 = jnp.int8
I32 = jnp.int32
ACC = jnp.float32          # matmuls accumulate in f32 whatever the operand dtype
CDTYPE = jnp.complex64     # only for the complex reference FFTs in experimental/fft/tricks.py
BYTES_PER_COMPONENT = {F: 4, BF: 2, I8: 1}   # f32->complex64, bf16->"complex32", int8->"complex16"

# A radix stage contracts over a whole 3-D block; the leaf is a plain 2-D matmul.
CBK_MIN_COLS = DEFAULT_ACCEL.transpose_fold_cols   # measured crossover for this unit

# Two spellings of one radix stage. 'cbk' makes the permutation the matmul's write pattern, so
# the transposed copy is never materialised. It pays only on wide stages; the threshold is
# empirically tested, and wall clock cannot resolve it -- it needs device time.
stage_matmul = lambda mat, x: jnp.einsum('pq,qst->pst', mat, x, preferred_element_type=ACC)
stage_matmul_t = lambda mat, x: jnp.einsum('pq,qst->spt', mat, x, preferred_element_type=ACC)
leaf_matmul = lambda mat, x: jnp.matmul(mat, x, preferred_element_type=ACC)


def dft_matrix(size, dtype=CDTYPE):
    """size x size COMPLEX DFT matrix W[r,c] = exp(-2i*pi*r*c/size)."""
    k = jnp.arange(size)
    return jnp.exp(-2j * jnp.pi * jnp.outer(k, k) / size).astype(dtype)


def dft_matrix_reim(size, dtype=F):
    """The same DFT matrix carried as real/imag instead of complex: (cos, sin, cos+sin).
    The third is precomputed because the Karatsuba product below needs it every time."""
    k = jnp.arange(size)
    angle = -2 * jnp.pi * jnp.outer(k, k) / size
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    return cos.astype(dtype), sin.astype(dtype), (cos + sin).astype(dtype)


def complex_matmul(matmul, cos, sin, cos_plus_sin, x_re, x_im):
    """(cos + i*sin) @ (x_re + i*x_im) in THREE real matmuls instead of four (Gauss/Karatsuba).
    `matmul` is the primitive to use — stage_matmul inside a radix stage, leaf_matmul at the leaf."""
    a = matmul(cos, x_re)
    b = matmul(sin, x_im)
    c = matmul(cos_plus_sin, x_re + x_im)
    return a - b, c - a - b                      # real part, imaginary part


def _twiddle(stage_len, n_cols):
    """Twiddle factors W_stage_len^(row*col) for a stage, as (cos, sin) broadcastable over batch."""
    if n_cols >= CBK_MIN_COLS:                       # (cols, radix) for the cbk stage output
        rows, cols = jnp.arange(RADIX)[None, :], jnp.arange(n_cols)[:, None]
    else:                                            # (radix, cols) for the bck one
        rows, cols = jnp.arange(RADIX)[:, None], jnp.arange(n_cols)[None, :]
    angle = (-2 * jnp.pi * rows * cols / stage_len).astype(F)
    return jnp.cos(angle)[:, :, None], jnp.sin(angle)[:, :, None]


def _undo_digit_reversal(out_re, out_im, stages, leaf_len):
    """The staged sweep leaves the bins in digit-reversed order; folding each stage back in
    reverse is pure reshaping (no data movement on TPU)."""
    length = leaf_len
    for radix in reversed(stages):
        out_re = out_re.reshape(length, radix, -1).reshape(length * radix, -1)
        out_im = out_im.reshape(length, radix, -1).reshape(length * radix, -1)
        length *= radix
    return out_re, out_im


def rfft_staged(signal, n, half=True, dtype=F):
    """Radix-RADIX FFT of REAL columns `signal` (n, batch), carried as real/imag in `dtype`.

    Peels one radix stage per iteration (length /RADIX, batch *RADIX), then a leaf DFT, then the
    reshape-only digit-reversal rebuild.  This is THE shared real pass — 1-D rfft and the axis-0
    half of the 2-D rfft both call it.  Returns (re, im)."""
    cur_re, cur_im = signal.astype(dtype), None
    length, batch, stages = n, signal.shape[1], []
    input_is_real = True                     # first stage has no imaginary part -> 2 matmuls, not 3

    while length > RADIX and length % RADIX == 0:
        n_cols = length // RADIX
        cos, sin, cos_plus_sin = dft_matrix_reim(RADIX, dtype)

        if input_is_real:
            block = cur_re.reshape(RADIX, n_cols, batch)
            mm = stage_matmul_t if n_cols >= CBK_MIN_COLS else stage_matmul
            stage_re, stage_im = mm(cos, block), mm(sin, block)
            input_is_real = False
        else:
            stage_re, stage_im = complex_matmul(
                stage_matmul_t if n_cols >= CBK_MIN_COLS else stage_matmul,
                cos, sin, cos_plus_sin,
                cur_re.reshape(RADIX, n_cols, batch), cur_im.reshape(RADIX, n_cols, batch))

        tw_cos, tw_sin = _twiddle(length, n_cols)        # f32 twiddle applied to f32 accumulators
        cur_re = stage_re * tw_cos - stage_im * tw_sin
        cur_im = stage_re * tw_sin + stage_im * tw_cos
        if n_cols < CBK_MIN_COLS:                    # narrow stage: transpose explicitly
            cur_re, cur_im = cur_re.transpose(1, 0, 2), cur_im.transpose(1, 0, 2)
        cur_re = cur_re.reshape(n_cols, RADIX * batch).astype(dtype)   # store back in `dtype`
        cur_im = cur_im.reshape(n_cols, RADIX * batch).astype(dtype)

        stages.append(RADIX)
        length, batch = n_cols, RADIX * batch

    cos, sin, cos_plus_sin = dft_matrix_reim(length, dtype)            # leaf DFT
    if input_is_real:                                                  # n <= RADIX: never staged
        out_re, out_im = leaf_matmul(cos, cur_re), leaf_matmul(sin, cur_re)
    else:
        out_re, out_im = complex_matmul(leaf_matmul, cos, sin, cos_plus_sin, cur_re, cur_im)

    out_re, out_im = _undo_digit_reversal(out_re, out_im, stages, length)
    if half:
        out_re, out_im = out_re[:n // 2], out_im[:n // 2]
    return out_re.astype(dtype), out_im.astype(dtype)


def fft_staged_complex(x_re, x_im, n):
    """Radix-RADIX FFT of COMPLEX columns (Karatsuba on every stage, including the first), full
    n-bin output.  Used by the packing schemes and the `reduced` Pallas kernel."""
    cur_re, cur_im = x_re, x_im
    length, batch, stages = n, x_re.shape[1], []

    while length > RADIX and length % RADIX == 0:
        n_cols = length // RADIX
        cos, sin, cos_plus_sin = dft_matrix_reim(RADIX)
        stage_re, stage_im = complex_matmul(
            stage_matmul_t if n_cols >= CBK_MIN_COLS else stage_matmul,
            cos, sin, cos_plus_sin,
            cur_re.reshape(RADIX, n_cols, batch), cur_im.reshape(RADIX, n_cols, batch))

        tw_cos, tw_sin = _twiddle(length, n_cols)
        nr, ni = stage_re * tw_cos - stage_im * tw_sin, stage_re * tw_sin + stage_im * tw_cos
        if n_cols < CBK_MIN_COLS:
            nr, ni = nr.transpose(1, 0, 2), ni.transpose(1, 0, 2)
        cur_re = nr.reshape(n_cols, RADIX * batch)
        cur_im = ni.reshape(n_cols, RADIX * batch)

        stages.append(RADIX)
        length, batch = n_cols, RADIX * batch

    cos, sin, cos_plus_sin = dft_matrix_reim(length)
    out_re, out_im = complex_matmul(leaf_matmul, cos, sin, cos_plus_sin, cur_re, cur_im)
    return _undo_digit_reversal(out_re, out_im, stages, length)


def reverse_rows(x):
    """Reverse the rows of x (rows, cols), rows a multiple of 128, in O(rows*128) via block
    matmuls.  Mosaic has no rev/flip/gather, so this is the VMEM-lowerable row reversal."""
    rows, cols = x.shape
    n_blocks = rows // 128
    idx = jnp.arange(128)
    flip_within = (idx[None, :] == (127 - idx[:, None])).astype(F)      # 128x128 anti-identity
    blocks = jnp.einsum('pq,gqw->gpw', flip_within, x.reshape(n_blocks, 128, cols))
    if n_blocks > 1:
        b = jnp.arange(n_blocks)
        flip_blocks = (b[None, :] == (n_blocks - 1 - b[:, None])).astype(F)
        blocks = jnp.einsum('gf,fpw->gpw', flip_blocks, blocks)
    return blocks.reshape(rows, cols)
