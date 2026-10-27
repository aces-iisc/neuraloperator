"""
1-D FFT engines that LOST the bench.  Kept because
results/REPORT.md §1-2 quotes their numbers; nothing in the main tree imports them.

  fft_iter / fft_recur      complex64 radix-B FFT (iterative stage-loop / recursive twin).
                            Superseded by fft.rfft1d.fft_reim, which does the same work in
                            real/imag and skips the complex64 assembly tax (10-17% of device time).
  jax_rfft_int8             int8 "complex16" direct DFT — a dtype-comparison datapoint.
  rfft_coupled / rfft_reduced real-input packing schemes (pair-two-signals / even-odd), both a
                            complex64 and a real/imag (_reim) flavour.  Measured 10-20% slower
                            than plain reim at N>=4096: the pack/unpack costs more than it saves.
  full_via_coupled/reduced  full-spectrum reconstruction from the Hermitian half.

The winners live in fft/rfft1d.py.
"""
import jax
import jax.numpy as jnp

from ...fft.core import (ACC, CDTYPE, I8, I32, F, RADIX as B,
                         complex_matmul as _karatsuba, dft_matrix,
                         fft_staged_complex as _fft_c)

C = CDTYPE  # local alias (complex64)

_MAT_TRICK = lambda mat, x: jnp.matmul(mat, x, preferred_element_type=jnp.float32)


# ── complex64 radix-B FFT (iterative + recursive): reference algorithms, NOT wired into any engine
#    (they build complex64; the bench uses the real/imag `fft_reim` above instead) ───────────────
def fft_iter(x):
    """Iterative radix-B FFT of each column of x (N, K)  (or a 1-D length-N vector)."""
    one_d = (x.ndim == 1)
    if one_d:
        x = x[:, None]
    N, K = x.shape
    cur, m, batch = x.astype(CDTYPE), N, K
    stages = []  # (B, N1) per stage, for the rebuild

    while m > B and m % B == 0:  # forward sweep: one radix-B stage per iteration
        N1 = m // B
        cur = cur.reshape(B, N1, batch)
        cur = jnp.einsum('br,rck->bck', dft_matrix(B), cur, preferred_element_type=CDTYPE)
        r = jnp.arange(B)[:, None]
        c = jnp.arange(N1)[None, :]
        cur = cur * jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]  # twiddle
        cur = cur.transpose(1, 0, 2).reshape(N1, B * batch)   # rotate c to axis0; fold b in
        stages.append((B, N1))
        m, batch = N1, B * batch

    cur = dft_matrix(m) @ cur   # leaf: one direct DFT over the remaining axis 0

    out, length = cur, m
    for (Bi, _N1i) in reversed(stages):  # rebuild: undo the folds in reverse order
        out = out.reshape(length, Bi, -1).reshape(length * Bi, -1)  # merge b_i back (outer)
        length = length * Bi
    return out[:, 0] if one_d else out


def fft_recur(x):
    """Recursive radix-B FFT of each column of x (N, K) (or 1-D length-N).  Recursive twin of
    fft_iter: radix-B stage -> twiddle -> transpose -> recurse on the length-N1 rows; leaf is a
    direct m-DFT matmul (m <= B or m % B != 0).  (Was small_dft_baseline in the archived code.)"""
    one_d = (x.ndim == 1)
    if one_d:
        x = x[:, None]
    out = _recur(x.astype(CDTYPE), x.shape[0])
    return out[:, 0] if one_d else out


def _recur(vecs, m):
    """Recursive radix-B DFT of the columns of vecs (m, K).  Leaf = direct m-DFT matmul."""
    if m <= B or m % B != 0:
        return jnp.matmul(dft_matrix(m), vecs, preferred_element_type=CDTYPE)
    N1, K = m // B, vecs.shape[1]
    Xb = vecs.reshape(B, N1, K)
    Yb = jnp.einsum('br,rck->bck', dft_matrix(B), Xb, preferred_element_type=CDTYPE)  # radix-B
    r = jnp.arange(B)[:, None]
    c = jnp.arange(N1)[None, :]
    tw = jnp.exp(-2j * jnp.pi * (r * c) / m).astype(CDTYPE)[:, :, None]  # twiddle W_m^(b*c)
    Zc = (Yb * tw).transpose(1, 0, 2).reshape(N1, B * K)  # twiddle, then transpose + fold b in
    Wd = _recur(Zc, N1)  # recurse on the length-N1 row DFT
    return Wd.reshape(N1, B, K).reshape(m, K)  # merge b back (outer)


@jax.jit
def fft(x):
    return fft_iter(x)


_MAT_I8 = lambda A, x: jnp.matmul(A, x, preferred_element_type=I32)  # int8 x int8 -> int32

def jax_rfft_int8(xr, half=True, scale=127.0):
    """Minimal 8-bit variant for the dtype-comparison case: a DIRECT (single-matmul) real DFT
    with symmetric-quantized int8 operands, int32 accumulate.  int8 CANNOT represent the DFT
    matrix by a plain .astype like bf16 (its entries live in [-1,1] and underflow to {-1,0,1}),
    so it needs a SCALE — that scale is the FFT-algorithm knob to tune (per-column input scale
    beats the per-tensor one below; a staged int8 radix would requant each stage).  This is a
    runnable starting point: fair on I/O bytes (1 byte/component) but O(N*rows) direct DFT, so
    keep N modest (OOMs like any direct DFT at large N).  Real (N, K) -> (yr, yi)."""
    N, K = xr.shape
    rows = N // 2 if half else N
    k, j = jnp.arange(rows)[:, None], jnp.arange(N)[None, :]
    ang = -2 * jnp.pi * k * j / N
    Cr = jnp.round(jnp.cos(ang) * scale).astype(I8)  # DFT matrix -> int8, scale 1/`scale`
    Ci = jnp.round(jnp.sin(ang) * scale).astype(I8)
    s = jnp.max(jnp.abs(xr)) / scale + 1e-9  # input symmetric scale (per-tensor; your knob)
    xq = jnp.round(xr / s).astype(I8)
    deq = s / scale  # dequant: undo both operand scales after the int32 matmul
    return _MAT_I8(Cr, xq).astype(F) * deq, _MAT_I8(Ci, xq).astype(F) * deq


# ── real-input packing tricks (parameterized by a complex-FFT engine) ─────────
ENGINES = {
    "jnpfft":   lambda z: jnp.fft.fft(z, axis=0),   # opaque custom-call (no fusion)
    "fft_iter": fft_iter,                            # our matmul FFT (fusable)
}


def _mirror(Z):
    """conj(Z[(M-k) mod M]) along axis 0 — the Hermitian partner of each bin."""
    return jnp.conj(jnp.roll(Z[::-1], 1, axis=0))


def rfft_base(xr, eng):
    """xr real (N, K) -> (N//2+1, K).  eng = complex FFT along axis 0."""
    N = xr.shape[0]
    return eng(xr.astype(C))[:N // 2 + 1]


def rfft_coupled(xr, eng, ratio=0.5):
    """Pair two real signals a,b -> z=a+ib, ONE length-N complex FFT, unpack the first
    m = round(N*ratio) Hermitian rows of each (m = n_modes for the FNO).  K must be even.
    NOTE: `eng` still computes the FULL N-bin FFT; `ratio` only trims the UNPACK.  Returns (m, K)."""
    N, K = xr.shape
    a, b = xr[:, :K // 2], xr[:, K // 2:]  # pair signals: a,b are (N, K/2)
    Z = eng((a + 1j * b).astype(C))  # ONE FFT on K/2 packed cols -> (N, K/2)
    _N = int(N * ratio)
    _N = _N if _N % 2 == 0 else _N + 1  # ensure even
    M_slice = jnp.conj(jnp.roll(Z[::-1], 1, axis=0)[:_N])  # Hermitian partner (reversed slice)
    A = 0.5 * (Z[:_N] + M_slice)     # rfft(a): (_N, K/2)
    Bb = -0.5j * (Z[:_N] - M_slice)  # rfft(b): (_N, K/2)
    return jnp.concatenate([A, Bb], axis=1)  # (_N, K)


def rfft_coupled_reim(xr, ratio=0.5, dt=F):
    """Trick A, but real/imag ALL THE WAY — no complex64, no `c64` custom-call.  Pack a,b -> z=a+ib,
    ONE complex FFT via the real/imag core `_fft_c` (returns (Zr,Zi)), unpack in real/imag with
    slice-reversal (gather-free), return (yr, yi).  Tests whether dropping the c64 assembly (~12%
    of the coupled scheme's device time) is a net win."""
    N, K = xr.shape
    a, b = xr[:, :K // 2], xr[:, K // 2:]
    Zr, Zi = _fft_c(a.astype(dt), b.astype(dt), N)
    m = int(N * ratio)
    m = m if m % 2 == 0 else m + 1
    Mr = jnp.concatenate([Zr[:1], Zr[:0:-1]], axis=0)[:m]      # Re Z[(N-k)%N], gather-free
    Mi = -jnp.concatenate([Zi[:1], Zi[:0:-1]], axis=0)[:m]     # -Im (conjugate)
    Zr, Zi = Zr[:m], Zi[:m]
    Ar, Ai = 0.5 * (Zr + Mr), 0.5 * (Zi + Mi)                  # rfft(a)
    Br, Bi = 0.5 * (Zi - Mi), -0.5 * (Zr - Mr)                 # rfft(b) = -j(Z-conjZ')/2
    return (jnp.concatenate([Ar, Br], axis=1).astype(dt),
            jnp.concatenate([Ai, Bi], axis=1).astype(dt))


def rfft_reduced_reim(xr, ratio=0.5, dt=F):
    """Trick B, real/imag ALL THE WAY — no complex64.  Even/odd -> z=even+i*odd, ONE HALF-length
    FFT via `_fft_c`, mirror via slice-reversal, twist X[k]=E[k]+W_N^k O[k] in real/imag, return
    (yr, yi).  No c64 assembly."""
    N, K = xr.shape
    h = N // 2
    x2 = xr.reshape(h, 2, K).astype(dt)
    Zr, Zi = _fft_c(x2[:, 0, :], x2[:, 1, :], h)              # half-length FFT of even+i*odd
    Mr = jnp.concatenate([Zr[:1], Zr[:0:-1]], axis=0)         # mirror M[k]=conj(Z[(h-k)%h])
    Mi = -jnp.concatenate([Zi[:1], Zi[:0:-1]], axis=0)
    Er, Ei = 0.5 * (Zr + Mr), 0.5 * (Zi + Mi)                 # even sub-DFT
    Or_, Oi = 0.5 * (Zi - Mi), -0.5 * (Zr - Mr)               # odd sub-DFT = -0.5j(Z-M)
    k = jnp.arange(h)
    twr = jnp.cos(-2 * jnp.pi * k / N).astype(F)[:, None]
    twi = jnp.sin(-2 * jnp.pi * k / N).astype(F)[:, None]
    yr = Er + (twr * Or_ - twi * Oi)                          # X[k]=E[k]+W_N^k O[k]
    yi = Ei + (twr * Oi + twi * Or_)
    return yr.astype(dt), yi.astype(dt)


def rfft_reduced(xr, eng, ratio=0.5):
    """Pack even/odd -> z=x_even+i*x_odd, ONE HALF-length FFT, last-layer twist
    X[k]=E[k]+W_N^k O[k].  Returns the first round(N*ratio) bins.  NOTE: `eng` still returns all
    N/2 bins; `ratio` only trims the twist/unpack output."""
    N, _K = xr.shape
    h = N * ratio
    h = int(h) if h % 2 == 0 else int(h) + 1  # ensure even
    z = (xr[0::2] + 1j * xr[1::2]).astype(C)  # (h, K)
    Z = eng(z)  # K HALF-length FFTs
    M = jnp.conj(jnp.concatenate([Z[:1], Z[:0:-1]], axis=0))  # mirror, gather-free
    E = 0.5 * (Z + M)
    O = -0.5j * (Z - M)
    rows = max(1, min(int(N * ratio), h + 1))  # needed Hermitian rows
    main_rows = min(rows, h)  # k in [0, h-1]: direct index, no modulo/gather
    kf = jnp.arange(main_rows)
    tw = jnp.exp(-2j * jnp.pi * kf / N).astype(C)[:, None]
    out_main = E[:main_rows] + tw * O[:main_rows]
    if rows == h + 1:  # k = h (Nyquist) maps to index 0 with tw = -1 (static branch)
        nyq = (E[0] - O[0])[None, :]
        return jnp.concatenate([out_main, nyq], axis=0)
    return out_main


_VARIANTS = {"base": rfft_base, "coupled": rfft_coupled, "reduced": rfft_reduced}


# ── full-spectrum (N, K) reconstruction from the half, vs recompute ──────────
def _reconstruct_full(half, N):
    """(N//2+1, K) Hermitian half -> full (N, K) by conjugate-mirroring the upper bins."""
    upper = jnp.conj(half[N // 2 - 1:0:-1])          # bins N/2+1..N-1 = conj(bins N/2-1..1)
    return jnp.concatenate([half, upper], axis=0)    # (N, K)


def full_recompute(xr, eng):
    """FULL (N,K): compute ALL N bins directly (plain complex FFT, no symmetry used)."""
    return eng(xr.astype(C))


def full_via_coupled(xr, eng):
    """FULL (N,K): coupled rfft (half work) + conjugate reconstruct."""
    return _reconstruct_full(rfft_coupled(xr, eng), xr.shape[0])


def rfft_coupled_low_modes(xr, n_modes, dt=F):
    """The coupled packing applied to a LOW-MODE transform: pair the K real signals into K/2
    complex ones, compute only the rows the unpack needs, then separate them.

    Recovering A[k] and B[k] needs both Z[k] and its mirror Z[(N-k) mod N], so a band of m modes
    forces 2m-1 rows of the DFT matrix rather than m.  That is the scheme's real cost and the band
    does not shrink it.

    cos and sin are stacked into one matmul rather than issued as Gauss/Karatsuba's three.  The
    three-matmul form trades a matmul for an elementwise pass, which is the right trade when the
    matmul dominates; at these shapes the matrix unit is a tenth of the call, so one wider call
    and one fewer pass is cheaper.  The packed halves are the left and right column blocks of xr,
    so [a | b] is xr itself and the operand costs nothing to form.

    Real (N, K), K even -> (re, im), each (n_modes, K).
    """
    n, k_cols = xr.shape
    cols = k_cols // 2

    low = jnp.arange(n_modes)
    mirror = (n - jnp.arange(1, n_modes)) % n
    rows = jnp.concatenate([low, mirror])          # 2m-1 rows, as the unpack needs
    n_rows = rows.shape[0]

    angle = -2.0 * jnp.pi * jnp.outer(rows, jnp.arange(n)).astype(F) / n
    stacked = jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=0).astype(dt)
    both = jnp.matmul(stacked, xr.astype(dt), preferred_element_type=ACC)   # [a | b] == xr

    cos_a, cos_b = both[:n_rows, :cols], both[:n_rows, cols:]
    sin_a, sin_b = both[n_rows:, :cols], both[n_rows:, cols:]
    z_re, z_im = cos_a - sin_b, sin_a + cos_b      # (cos + i sin) @ (a + i b)

    zr, zi = z_re[:n_modes], z_im[:n_modes]
    mr = jnp.concatenate([z_re[:1], z_re[n_modes:]], axis=0)
    mi = jnp.concatenate([z_im[:1], z_im[n_modes:]], axis=0)
    a_re, a_im = 0.5 * (zr + mr), 0.5 * (zi - mi)
    b_re, b_im = 0.5 * (zi + mi), -0.5 * (zr - mr)
    return (jnp.concatenate([a_re, b_re], axis=1).astype(dt),
            jnp.concatenate([a_im, b_im], axis=1).astype(dt))

def rfft_reduced_low_modes(xr, n_modes, dt=F):
    """The reduced split applied to a LOW-MODE transform.
    Even and odd samples are transformed separately at length N/2 and recombined with the twiddle,
    X[k] = E[k] + W_N^k O[k].  A band of m modes needs E and O at m rows only, so both halves are
    partial transforms.

    The arithmetic does not improve on a plain partial DFT: two m x (N/2) matrices hold the same
    m*N entries as one m x N matrix, and the recombination is added on top.  Folding the twiddle
    into the odd matrix turns the pair back into exactly the partial DFT's matrix, so the split's
    only contribution at a band is work it then has to undo.

    What is avoidable is how it is issued.  A row-major (N, K) array reshaped to (N/2, 2K) already
    has [even_i | odd_i] side by side in row i, so the operand needs no stride-2 gather, and
    stacking cos over sin gives all four products from one (2m, N/2) x (N/2, 2K) call instead of
    four narrow ones.

    Real (N, K), N even -> (re, im), each (n_modes, K).
    """
    n, k_cols = xr.shape
    half = n // 2
    operand = xr.reshape(half, 2 * k_cols)        # row i = [even_i | odd_i], no copy

    rows = jnp.arange(n_modes)
    angle = -2.0 * jnp.pi * jnp.outer(rows, jnp.arange(half)).astype(F) / half
    stacked = jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=0).astype(dt)
    both = jnp.matmul(stacked, operand, preferred_element_type=ACC)

    e_re, o_re = both[:n_modes, :k_cols], both[:n_modes, k_cols:]
    e_im, o_im = both[n_modes:, :k_cols], both[n_modes:, k_cols:]
    tw = -2.0 * jnp.pi * rows.astype(F) / n
    wr, wi = jnp.cos(tw)[:, None], jnp.sin(tw)[:, None]
    return e_re + wr * o_re - wi * o_im, e_im + wr * o_im + wi * o_re


def full_via_reduced(xr, eng):
    """FULL (N,K): reduced rfft (half work) + conjugate reconstruct."""
    return _reconstruct_full(rfft_reduced(xr, eng), xr.shape[0])


_FULL = {"recompute": full_recompute,
         "viacoupled": full_via_coupled,
         "viareduced": full_via_reduced}








