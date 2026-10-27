"""Real-input DFT primitives, carried as real/imaginary pairs throughout.

No complex tensor appears anywhere: the matrix unit has no complex multiply, and a complex
dtype costs a wrapper around the same real matmuls plus a layout the contraction cannot use.
"""
import functools
import os

import torch

# einsum index letters, used with these meanings everywhere in this package:
#   n  sample          i  in-channel      o  out-channel    c  channel (in == out)
#   h  grid row        w  grid col        a  row mode (the 2m band)   d  col mode (the low m)
#   r  CP rank
# The staged transform works on different tensors, so it uses its own letters rather than
# borrowing these with new meanings:
#   p  stage output    q  stage input (the radix)   s  stage column   t  packed batch
#   g  block group (the row reversal)

from ..arch import detect

# TF32 is the default here, which is the opposite of the TPU setting this was ported from.
# On a TPU "highest" precision is free: the layer is memory-bound and the extra passes of a
# true-f32 matmul hide behind the HBM wait.  On this hardware an f32 matmul does not reach the
# tensor cores at all -- ncu puts sm__pipe_tensor_cycles_active at 0.00% for every simt and
# MAGMA kernel, against 34.78% for the tensorop ones -- so "highest" quietly spends the tensor
# cores to buy accuracy nothing needs.  TF32 keeps f32 range with a 10-bit mantissa and costs
# about 1e-3 relative, which is well inside the transform's own f32 round-off.
torch.set_float32_matmul_precision(os.environ.get("FNO_MATMUL_PRECISION", "high"))
torch.backends.cuda.matmul.allow_tf32 = os.environ.get("FNO_MATMUL_PRECISION", "high") != "highest"
torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32

DEFAULT_ACCEL = detect()
RADIX = DEFAULT_ACCEL.radix
F = torch.float32
BF = torch.bfloat16
ACC = torch.float32              # matmuls accumulate in f32 whatever the operands are
CBK_MIN_COLS = DEFAULT_ACCEL.transpose_fold_cols

_CACHE_LIMIT = int(os.environ.get("FNO_MATRIX_CACHE", "256"))


def _mm(a, b):
    """Matmul in the operand dtype, result in f32.

    torch has no `preferred_element_type`, and a bf16 matmul returns bf16 even though the
    tensor cores accumulated in f32.  Upcasting on the way out keeps the accumulator width the
    rest of the layer assumes; the operands stay narrow, which is where the traffic is saved.
    """
    return torch.matmul(a, b).to(ACC)


def _einsum(spec, a, b):
    return torch.einsum(spec, a, b).to(ACC)


@functools.lru_cache(maxsize=_CACHE_LIMIT)
def _angle_key(size, dtype, device):
    """cos/sin of a full size x size DFT. Cached: in eager mode this is rebuilt per call."""
    k = torch.arange(size, device=device, dtype=F)
    angle = -2 * torch.pi * torch.outer(k, k) / size
    cos, sin = torch.cos(angle), torch.sin(angle)
    return cos.to(dtype), sin.to(dtype), (cos + sin).to(dtype)


def dft_matrix_reim(size, dtype=F, device=None):
    """(cos, sin, cos+sin) of a size x size DFT. The third is what Karatsuba needs."""
    return _angle_key(size, dtype, torch.device(device or "cpu"))


@functools.lru_cache(maxsize=_CACHE_LIMIT)
def _rows_matrix(rows, length, sign, dtype, device):
    """cos and sin stacked along the output axis, for the DFT restricted to `rows`."""
    k = torch.tensor(rows, device=device, dtype=F).unsqueeze(1)
    n = torch.arange(length, device=device, dtype=F).unsqueeze(0)
    angle = (sign * 2.0 * torch.pi / length) * (k * n)
    return torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)


def rows_matrix(rows, length, sign=-1, dtype=F, device=None):
    key = tuple(int(r) for r in rows)
    return _rows_matrix(key, length, sign, dtype, torch.device(device or "cpu"))


def stacked_rows(rows, length, sign=-1, dtype=F, device=None):
    cos, sin = rows_matrix(rows, length, sign, dtype, device)
    return torch.cat([cos, sin], dim=0)


def complex_matmul(matmul, cos, sin, cos_plus_sin, x_re, x_im):
    """(cos + i sin) @ (x_re + i x_im) in three real matmuls instead of four."""
    a = matmul(cos, x_re)
    b = matmul(sin, x_im)
    c = matmul(cos_plus_sin, x_re + x_im)
    return a - b, c - a - b


def _stage_matmul(mat, x):
    return _einsum("pq,qst->pst", mat, x)


def _stage_matmul_t(mat, x):
    """The permutation becomes the matmul's write pattern, so no transposed copy is built."""
    return _einsum("pq,qst->spt", mat, x)


def _fold_transpose(n_cols):
    return CBK_MIN_COLS is not None and n_cols >= CBK_MIN_COLS


@functools.lru_cache(maxsize=_CACHE_LIMIT)
def _twiddle(stage_len, n_cols, radix, folded, device):
    if folded:
        rows = torch.arange(radix, device=device, dtype=F).unsqueeze(0)
        cols = torch.arange(n_cols, device=device, dtype=F).unsqueeze(1)
    else:
        rows = torch.arange(radix, device=device, dtype=F).unsqueeze(1)
        cols = torch.arange(n_cols, device=device, dtype=F).unsqueeze(0)
    angle = -2 * torch.pi * rows * cols / stage_len
    return torch.cos(angle).unsqueeze(-1), torch.sin(angle).unsqueeze(-1)


def _undo_digit_reversal(out_re, out_im, stages, leaf_len):
    length = leaf_len
    for radix in reversed(stages):
        out_re = out_re.reshape(length, radix, -1).reshape(length * radix, -1)
        out_im = out_im.reshape(length, radix, -1).reshape(length * radix, -1)
        length *= radix
    return out_re, out_im


def rfft_staged(signal, n, half=True, dtype=F):
    """Radix-RADIX FFT of real columns (n, batch), carried as real/imag. Returns (re, im)."""
    device = signal.device
    cur_re, cur_im = signal.to(dtype), None
    length, batch, stages = n, signal.shape[1], []
    input_is_real = True                 # first stage has no imaginary part: 2 matmuls, not 3

    while length > RADIX and length % RADIX == 0:
        n_cols = length // RADIX
        folded = _fold_transpose(n_cols)
        mm = _stage_matmul_t if folded else _stage_matmul
        cos, sin, cos_plus_sin = dft_matrix_reim(RADIX, dtype, device)

        if input_is_real:
            block = cur_re.reshape(RADIX, n_cols, batch)
            stage_re, stage_im = mm(cos, block), mm(sin, block)
            input_is_real = False
        else:
            stage_re, stage_im = complex_matmul(
                mm, cos, sin, cos_plus_sin,
                cur_re.reshape(RADIX, n_cols, batch), cur_im.reshape(RADIX, n_cols, batch))

        tw_cos, tw_sin = _twiddle(length, n_cols, RADIX, folded, device)
        cur_re = stage_re * tw_cos - stage_im * tw_sin
        cur_im = stage_re * tw_sin + stage_im * tw_cos
        if not folded:
            cur_re, cur_im = cur_re.permute(1, 0, 2), cur_im.permute(1, 0, 2)
        cur_re = cur_re.reshape(n_cols, RADIX * batch).to(dtype)
        cur_im = cur_im.reshape(n_cols, RADIX * batch).to(dtype)

        stages.append(RADIX)
        length, batch = n_cols, RADIX * batch

    cos, sin, cos_plus_sin = dft_matrix_reim(length, dtype, device)
    if input_is_real:
        out_re, out_im = _mm(cos, cur_re), _mm(sin, cur_re)
    else:
        out_re, out_im = complex_matmul(_mm, cos, sin, cos_plus_sin, cur_re, cur_im)

    out_re, out_im = _undo_digit_reversal(out_re, out_im, stages, length)
    if half:
        out_re, out_im = out_re[:n // 2], out_im[:n // 2]
    return out_re.to(dtype), out_im.to(dtype)


def fft_staged_complex(x_re, x_im, n):
    """Radix-RADIX FFT of complex columns, full n-bin output."""
    device = x_re.device
    cur_re, cur_im = x_re, x_im
    length, batch, stages = n, x_re.shape[1], []

    while length > RADIX and length % RADIX == 0:
        n_cols = length // RADIX
        folded = _fold_transpose(n_cols)
        cos, sin, cos_plus_sin = dft_matrix_reim(RADIX, x_re.dtype, device)
        stage_re, stage_im = complex_matmul(
            _stage_matmul_t if folded else _stage_matmul, cos, sin, cos_plus_sin,
            cur_re.reshape(RADIX, n_cols, batch), cur_im.reshape(RADIX, n_cols, batch))

        tw_cos, tw_sin = _twiddle(length, n_cols, RADIX, folded, device)
        nr = stage_re * tw_cos - stage_im * tw_sin
        ni = stage_re * tw_sin + stage_im * tw_cos
        if not folded:
            nr, ni = nr.permute(1, 0, 2), ni.permute(1, 0, 2)
        cur_re, cur_im = nr.reshape(n_cols, RADIX * batch), ni.reshape(n_cols, RADIX * batch)

        stages.append(RADIX)
        length, batch = n_cols, RADIX * batch

    cos, sin, cos_plus_sin = dft_matrix_reim(length, x_re.dtype, device)
    out_re, out_im = complex_matmul(_mm, cos, sin, cos_plus_sin, cur_re, cur_im)
    return _undo_digit_reversal(out_re, out_im, stages, length)
