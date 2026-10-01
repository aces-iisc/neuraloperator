"""
EFFICIENCY VARIANTS of the reim FNO, for
research only.  The production model stays in fno/model.py; nothing here is imported by it.

Implements a few of the literature techniques (except the fused FFT-GEMM-iFFT kernel, which is being
done separately), all keeping the spectral transform real/imag (no complex64):

  • bf16-activations forward      — carry activations in bf16 (half the activation HBM) to test the
                                    throughput win a memory-bound FNO should get (weights-only bf16
                                    did NOT help — activations dominate the traffic).
  • separable spectral weights    — F-FNO style: one 1-D spectral weight per axis, summed.
                                    params per layer: 2·Cin·Cout·m   (vs   Cin·Cout·m²).
  • CP-factorized spectral weights— TFNO style: rank-R CP decomposition of the (Cin,Cout,m,m) weight.
                                    params per layer: 2·R·(Cin+Cout+2m)  (re & im).
  • incremental modes             — curriculum that unmasks high modes over training.

The forward is ONE function `forward(P, x, spectral)`; `spectral` plugs in the variant filter.
Run:  python tpu/experimental/fno_variants.py     (CPU demo: params + Darcy16 rel L2)
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from ...fft.core import ACC, BF, F
from ...fno.data import load_darcy
from ...fno.layers import (_init_conv1x1 as _conv1x1, _init_mlp as _mlp, add_grid,
                           adam_init as _adam_init, adam_step as _adam_step,
                           channel_mlp, conv1x1, mse, rel_l2)
from .onecorner import _dft, fft2_lowmodes, ifft2_real

gelu = jax.nn.gelu
_e = lambda M, Z, s="": jnp.einsum(s, M, Z, preferred_element_type=ACC)


# ═══════════════════════════════════════════════════════════════════════════
#  Generic forward — `spectral(filt, fr, fi) -> (gr, gi)` selects the variant.
# ═══════════════════════════════════════════════════════════════════════════
def forward(P, x, spectral, m):
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        fr, fi = fft2_lowmodes(x, m)                         # partial DFT -> low modes (re, im)
        gr, gi = spectral(blk["filter"], fr, fi)             # variant complex channel-mix
        spec = ifft2_real(gr, gi, x.shape[-2], x.shape[-1])  # inverse -> real field
        x = gelu(spec + conv1x1(blk["skip"], x))             # + skip, activate
    return channel_mlp(P["proj"], x)


# ── (baseline) dense spectral filter, for a fair params/accuracy comparison ──
def dense_filter(w, fr, fi):
    mix = lambda M, Z: jnp.einsum("iokl,bikl->bokl", M, Z, preferred_element_type=ACC)
    return mix(w["w_re"], fr) - mix(w["w_im"], fi), mix(w["w_re"], fi) + mix(w["w_im"], fr)


def dense_init(key, cin, cout, m):
    s = (2.0 / (cin + cout)) ** 0.5
    kr, ki = jax.random.split(key)
    return {"w_re": s * jax.random.normal(kr, (cin, cout, m, m)),
            "w_im": s * jax.random.normal(ki, (cin, cout, m, m))}


# ══════════ VARIANT 1 — separable (F-FNO): a 1-D weight per axis, summed ══════════
def sep_filter(w, fr, fi):
    """g = (W_k ⊛ f) + (W_l ⊛ f):  W_k depends on the k-axis mode, W_l on the l-axis mode."""
    def mix(Wk, Wl, Z):
        gk = jnp.einsum("iok,bikl->bokl", Wk, Z, preferred_element_type=ACC)   # weight per k
        gl = jnp.einsum("iol,bikl->bokl", Wl, Z, preferred_element_type=ACC)   # weight per l
        return gk + gl
    gr = mix(w["kr"], w["lr"], fr) - mix(w["ki"], w["li"], fi)
    gi = mix(w["kr"], w["lr"], fi) + mix(w["ki"], w["li"], fr)
    return gr, gi


def sep_init(key, cin, cout, m):
    s = (2.0 / (cin + cout)) ** 0.5
    ks = jax.random.split(key, 4)
    shp = (cin, cout, m)
    return {"kr": s * jax.random.normal(ks[0], shp), "ki": s * jax.random.normal(ks[1], shp),
            "lr": s * jax.random.normal(ks[2], shp), "li": s * jax.random.normal(ks[3], shp)}


# ══════════ VARIANT 2 — CP-factorized (TFNO): rank-R decomposition of (Cin,Cout,m,m) ══════════
def _cp_mix(A, B, C, D, f):
    """Contract f (B,Cin,m,m) with the rank-R CP weight A[i,r]B[o,r]C[k,r]D[l,r] -> (B,Cout,m,m)."""
    t = jnp.einsum("ir,bikl->brkl", A, f, preferred_element_type=ACC)   # contract Cin
    t = t * C.T[None, :, :, None] * D.T[None, :, None, :]               # apply C[k,r], D[l,r]
    return jnp.einsum("or,brkl->bokl", B, t, preferred_element_type=ACC)  # contract R -> Cout


def cp_filter(w, fr, fi):
    m1 = lambda p: _cp_mix(w[p + "A"], w[p + "B"], w[p + "C"], w[p + "D"], fr)
    m2 = lambda p: _cp_mix(w[p + "A"], w[p + "B"], w[p + "C"], w[p + "D"], fi)
    return m1("re_") - m2("im_"), m1("im_") + m2("re_")


def cp_init(key, cin, cout, m, rank=8):
    s = (1.0 / (cin + cout + 2 * m)) ** 0.5
    d = {}
    ks = jax.random.split(key, 8)
    for j, part in enumerate(("re_", "im_")):
        d[part + "A"] = s * jax.random.normal(ks[4 * j + 0], (cin, rank))
        d[part + "B"] = s * jax.random.normal(ks[4 * j + 1], (cout, rank))
        d[part + "C"] = s * jax.random.normal(ks[4 * j + 2], (m, rank))
        d[part + "D"] = s * jax.random.normal(ks[4 * j + 3], (m, rank))
    return d


# ── model builder (variant-agnostic) ────────────────────────────────────────
def build(key, filt_init, in_ch=1, out_ch=1, width=24, m=16, n_layers=4, **fkw):
    ks = jax.random.split(key, n_layers + 2)
    return {"lift": _mlp(ks[0], in_ch + 2, 2 * width, width),
            "blocks": [{"filter": filt_init(jax.random.fold_in(ks[i + 1], 0), width, width, m, **fkw),
                        "skip": _conv1x1(jax.random.fold_in(ks[i + 1], 1), width, width)}
                       for i in range(n_layers)],
            "proj": _mlp(ks[-1], width, 2 * width, out_ch)}


# ══════════ VARIANT 3 — full bf16 activations (the throughput lever) ══════════
def forward_bf16(P, x, spectral, m):
    """Same graph as `forward`, but activations are carried in bf16 (matmuls accumulate f32 then
    cast to bf16) — halves the activation HBM traffic that gates a memory-bound FNO.  Weights are
    cast to bf16 too.  Needs a TPU run to see the throughput win (CPU won't show it)."""
    Pb = jax.tree_util.tree_map(lambda a: a.astype(BF), P)
    xb = x.astype(BF)
    y = forward(Pb, xb, spectral, m)                          # helpers accumulate f32
    return y.astype(BF)                                       # keep the output bf16


# ══════════ VARIANT 6 — jnp.rfft2 spectral conv (the "standard" full-FFT path, for comparison) ══════════
def forward_jnp(P, x, m):
    """FNO forward whose spectral conv uses jnp.fft.rfft2 (full FFT + truncate + irfft2, complex64)
    instead of the partial DFT — the standard neuralop path.  Compare vs `forward` (partial)."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        B, C, H, W = x.shape
        Xf = jnp.fft.rfft2(x, axes=(-2, -1))                 # (B,C,H,W//2+1) complex64 — full FFT
        Xlow = Xf[..., :m, :m]                               # keep low m×m corner
        w = blk["filter"]
        Wc = w["w_re"] + 1j * w["w_im"]
        Olow = jnp.einsum("iokl,bikl->bokl", Wc.astype(Xlow.dtype), Xlow)
        Cout = w["w_re"].shape[1]
        full = jnp.zeros((B, Cout, H, W // 2 + 1), Xlow.dtype).at[..., :m, :m].set(Olow)
        spec = jnp.fft.irfft2(full, s=(H, W), axes=(-2, -1))  # back to real
        x = gelu(spec + conv1x1(blk["skip"], x))
    return channel_mlp(P["proj"], x)


# ══════════ helpers: param count + a tiny trainer that takes a forward fn ══════════
def n_params(P):
    return int(sum(a.size for a in jax.tree_util.tree_leaves(P)))


def infer_us_per_sample(fwd, P, x, reps=30):
    """Inference time per sample for a jitted forward (compile/warmup excluded)."""
    import time
    fn = jax.jit(fwd)
    jax.block_until_ready(fn(P, x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(P, x))
        ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts) // 2]
    return med / x.shape[0] * 1e6, x.shape[0] / med          # (us/sample, samples/s)


def quick_train(fwd, P, xtr, ytr, xte, yte, steps=60, batch=32, lr=5e-3):
    xtr, ytr, xte, yte = map(jnp.asarray, (xtr, ytr, xte, yte))
    opt = _adam_init(P)

    @jax.jit
    def step(P, opt, x, y):
        loss, g = jax.value_and_grad(lambda P: mse(fwd(P, x), y))(P)
        P, opt = _adam_step(P, g, opt, lr=lr, weight_decay=1e-4)
        return P, opt, loss
    key = jax.random.PRNGKey(0)
    nb = max(1, xtr.shape[0] // batch)
    for ep in range(steps):
        key, ks = jax.random.split(key)
        perm = jax.random.permutation(ks, xtr.shape[0])
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            P, opt, _ = step(P, opt, xtr[idx], ytr[idx])
    return float(rel_l2(fwd(P, xte), yte))


# ══════════ VARIANT 4 — incremental modes (curriculum) ══════════
def _mode_mask(m, ma):
    """(m,m) 0/1 mask keeping only the lowest `ma` modes on each axis.  `ma` may be traced."""
    keep = (jnp.arange(m) < ma).astype(F)
    return keep[:, None] * keep[None, :]


def forward_incr(P, x, spectral, m, ma):
    """Like `forward` but only the lowest `ma`×`ma` modes are active (high modes zeroed).  Grow
    `ma` over training for a cheap-early / full-capacity-late curriculum."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    mask = _mode_mask(m, ma)[None, None]                     # (1,1,m,m)
    for blk in P["blocks"]:
        fr, fi = fft2_lowmodes(x, m)
        fr, fi = fr * mask, fi * mask                        # keep only low modes
        gr, gi = spectral(blk["filter"], fr, fi)
        spec = ifft2_real(gr, gi, x.shape[-2], x.shape[-1])
        x = gelu(spec + conv1x1(blk["skip"], x))
    return channel_mlp(P["proj"], x)


def train_incremental(P, xtr, ytr, xte, yte, m, steps=60, batch=32, lr=5e-3, m0=2, warm=0.7):
    """Grow active modes ma: m0 -> m over the first `warm` fraction of training, then hold at m."""
    xtr, ytr, xte, yte = map(jnp.asarray, (xtr, ytr, xte, yte))
    opt = _adam_init(P)

    @jax.jit
    def step(P, opt, x, y, ma):
        loss, g = jax.value_and_grad(lambda P: mse(forward_incr(P, x, dense_filter, m, ma), y))(P)
        P, opt = _adam_step(P, g, opt, lr=lr, weight_decay=1e-4)
        return P, opt, loss
    key = jax.random.PRNGKey(0)
    nb = max(1, xtr.shape[0] // batch)
    for ep in range(steps):
        frac = min(1.0, ep / (steps * warm))
        ma = jnp.float32(m0 + frac * (m - m0))              # traced -> no recompile
        key, ks = jax.random.split(key)
        perm = jax.random.permutation(ks, xtr.shape[0])
        for b in range(nb):
            idx = perm[b * batch:(b + 1) * batch]
            P, opt, _ = step(P, opt, xtr[idx], ytr[idx], ma)
    return float(rel_l2(forward_incr(P, xte, dense_filter, m, jnp.float32(m)), yte))


# ══════════ VARIANT 5 — layout / einsum-reorder (reduce corner-turn transposes) ══════════
def dense_filter_opt(w, fr, fi):
    """Dense spectral mix, but let XLA pick the optimal contraction path (optimize='optimal')."""
    mix = lambda M, Z: jnp.einsum("iokl,bikl->bokl", M, Z, preferred_element_type=ACC, optimize="optimal")
    return mix(w["w_re"], fr) - mix(w["w_im"], fi), mix(w["w_re"], fi) + mix(w["w_im"], fr)


def fft2_lowmodes_r(x, m):
    """Reordered forward partial-DFT: keep (b,c) as the leading pair and contract the spatial axes
    so the intermediate stays (b,c,·,·) contiguous — aims to avoid the relayout copy XLA inserts
    for the 'bckw' intermediate in the baseline."""
    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    ChR, ChI = _dft(m, H, -1)
    # contract W first (rightmost, contiguous), then H — reversed axis order vs the baseline
    CwR, CwI = _dft(m, W, -1)
    Br = jnp.einsum("jw,bchw->bchj", CwR, x, preferred_element_type=ACC)
    Bi = jnp.einsum("jw,bchw->bchj", CwI, x, preferred_element_type=ACC)  # real input -> imag from sin
    e = lambda M, Zr, Zi: (jnp.einsum("kh,bchj->bckj", M, Zr, preferred_element_type=ACC),
                           jnp.einsum("kh,bchj->bckj", M, Zi, preferred_element_type=ACC))
    Rr, Ri = e(ChR, Br, Bi)                                  # cos on (Br,Bi)
    Sr, Si = e(ChI, Br, Bi)                                  # sin on (Br,Bi)
    return Rr - Si, Ri + Sr                                  # (b,c,k,j) real/imag


def forward_reorder(P, x, spectral, m):
    """`forward` with the reordered forward DFT + optimal-path spectral mix."""
    x = add_grid(x)
    x = channel_mlp(P["lift"], x)
    for blk in P["blocks"]:
        fr, fi = fft2_lowmodes_r(x, m)
        gr, gi = spectral(blk["filter"], fr, fi)
        spec = ifft2_real(gr, gi, x.shape[-2], x.shape[-1])
        x = gelu(spec + conv1x1(blk["skip"], x))
    return channel_mlp(P["proj"], x)


def demo(npz=None, root="~/data"):
    if npz:
        d = np.load(npz); xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    else:
        xtr, ytr, xte, yte = load_darcy(os.path.expanduser(root), 16)
    key = jax.random.PRNGKey(0)
    W, M, L = 24, 8, 4
    print(f"Darcy16  width={W} modes={M} layers={L}  (60-step CPU demo — accuracy indicative)\n")
    print(f"  {'variant':<22}{'params':>10}{'rel L2':>10}")
    configs = [
        ("dense (baseline)", dense_init, dense_filter, {}),
        ("separable (F-FNO)", sep_init, sep_filter, {}),
        ("CP rank-4 (TFNO)", cp_init, cp_filter, {"rank": 4}),
        ("CP rank-8 (TFNO)", cp_init, cp_filter, {"rank": 8}),
    ]
    for name, init, filt, fkw in configs:
        P = build(key, init, width=W, m=M, n_layers=L, **fkw)
        rl = quick_train(lambda P, x, f=filt: forward(P, x, f, M), P, xtr, ytr, xte, yte)
        print(f"  {name:<22}{n_params(P):>10,}{rl:>10.4f}")

    # ── PAIRED bf16 vs f32 accuracy (same seed/data) — does bf16 help or hurt accuracy? ──
    print("\nbf16 vs f32 accuracy (dense filter, same seed) — does bf16 change accuracy?")
    P0 = build(key, dense_init, width=W, m=M, n_layers=L)
    rl_f32 = quick_train(lambda P, x: forward(P, x, dense_filter, M), P0, xtr, ytr, xte, yte)
    P0 = build(key, dense_init, width=W, m=M, n_layers=L)
    rl_bf16 = quick_train(lambda P, x: forward_bf16(P, x, dense_filter, M), P0, xtr, ytr, xte, yte)
    print(f"  {'f32 activations':<22}{'':>10}{rl_f32:>10.4f}")
    print(f"  {'bf16 activations':<22}{'':>10}{rl_bf16:>10.4f}   (delta {rl_bf16 - rl_f32:+.4f})")

    # ── incremental modes (curriculum: grow ma 2->M) ──
    print("\nincremental modes (grow ma 2->M over training) vs fixed-M dense")
    Pi = build(key, dense_init, width=W, m=M, n_layers=L)
    rl_incr = train_incremental(Pi, xtr, ytr, xte, yte, m=M)
    print(f"  {'incremental (2->'+str(M)+')':<22}{'':>10}{rl_incr:>10.4f}   (vs fixed {rl_f32:.4f})")

    # ── layout / einsum-reorder: correctness + inference time vs baseline ──
    print("\nlayout/einsum-reorder (32x32, batch 512) — device time vs baseline (accuracy-neutral)")
    Pr = build(key, dense_init, width=W, m=16, n_layers=L)
    xr = jax.random.normal(jax.random.PRNGKey(7), (512, 1, 32, 32))
    a = jnp.asarray(forward(Pr, xr[:2], dense_filter, 16))
    b = jnp.asarray(forward_reorder(Pr, xr[:2], dense_filter_opt, 16))
    print(f"  reorder matches baseline: rel_err {float(jnp.max(jnp.abs(a-b))/(jnp.max(jnp.abs(a))+1e-9)):.1e}")
    for tag, fwd in [("baseline layout", lambda P, x: forward(P, x, dense_filter, 16)),
                     ("reordered layout", lambda P, x: forward_reorder(P, x, dense_filter_opt, 16))]:
        us, sps = infer_us_per_sample(fwd, Pr, xr)
        print(f"  {tag:<22}{us:>8.2f} us/sample   {sps:>12,.0f} samples/s")

    # ── the throughput question: f32 activations vs bf16 activations (memory-bound) ──
    print("\nthroughput @ 32x32, batch 512  (bf16 ACTIVATIONS vs f32 — memory-bound test)")
    Pb = build(key, dense_init, width=W, m=16, n_layers=L)
    xb = jax.random.normal(jax.random.PRNGKey(7), (512, 1, 32, 32))
    for tag, fwd in [("f32 activations", lambda P, x: forward(P, x, dense_filter, 16)),
                     ("bf16 activations", lambda P, x: forward_bf16(P, x, dense_filter, 16))]:
        us, sps = infer_us_per_sample(fwd, Pb, xb)
        print(f"  {tag:<22}{us:>8.2f} us/sample   {sps:>12,.0f} samples/s")


if __name__ == "__main__":
    demo()
