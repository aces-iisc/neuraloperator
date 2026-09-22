"""
fno_reim.py — a Fourier Neural Operator in pure JAX whose spectral transform is OUR `reim`
real-FFT (fft.rfft1d.rfft_half): real input -> (real, imag) half
spectrum, no complex64, no c64 assembly.  This is the reim engine that won the v5e sweep, wired
into a full FNO.

Layers (functional; params are plain pytrees, so jax.grad/jit/Adam just work):
    spectral_conv   -> reim rfft -> keep low modes -> complex channel-mix -> reim inverse (IDFT
                       matmul from the kept modes) -> real field                    (the "R" operator)
    linear1x1       -> pointwise 1x1 channel mix                                    (skip / lift / project)
    channel_mlp     -> linear1x1 -> act -> linear1x1
    fno_block       -> act( spectral_conv(x) + linear1x1_skip(x) )                  (ONE Fourier layer)
    fno             -> lift -> [fno_block]*L -> project                             (the whole model)

The forward transform is reim (staged radix-B, real/imag).  The inverse only has to reconstruct
from the m KEPT modes, so it is a small real/imag partial-IDFT matmul (N x m) — itself reim-style
(real/imag, no complex64).  Data gen may use jnp.fft (targets only, not the model).

Run:  python tpu/fno_reim.py
"""
import jax
import jax.numpy as jnp

from ...fft.core import ACC, F
from ...fft.rfft1d import rfft_half


#  Layer 1 — spectral_conv   (the Fourier "R" operator, built on reim)
def spectral_conv_init(key, in_ch, out_ch, n_modes):
    """Complex weight R (in_ch, out_ch, n_modes), stored as separate real+imag (reim-style)."""
    std = (2.0 / (in_ch + out_ch)) ** 0.5
    kr, ki = jax.random.split(key)
    shape = (in_ch, out_ch, n_modes)
    return {"w_re": std * jax.random.normal(kr, shape),
            "w_im": std * jax.random.normal(ki, shape)}


def idft_lowmode(m, N, dt=F):
    """Real/imag partial-IDFT matrices (N, m): reconstruct a real field from its m LOW rfft modes.
      u[n] = (1/N)[ X0 + 2*sum_{k>=1}(Xr cos - Xi sin) ] = Cr[n,k]·Xr[k] - Ci[n,k]·Xi[k].
    The 1/N and the factor-2 (all non-DC bins fold their Hermitian partner) are baked into Cr,Ci."""
    n = jnp.arange(N)[:, None]
    k = jnp.arange(m)[None, :]
    ang = 2.0 * jnp.pi * n * k / N
    w = jnp.where(k == 0, 1.0, 2.0) / N        # DC once, others twice (Hermitian fold)
    return (jnp.cos(ang) * w).astype(dt), (jnp.sin(ang) * w).astype(dt)


def spectral_conv_apply(p, x, dt=F):
    """x : (B, in_ch, N) real -> (B, out_ch, N) real.  Forward = reim rfft; keep m low modes;
    complex channel-mix; inverse = reim-style partial-IDFT matmul."""
    m = p["w_re"].shape[-1]
    B, Cin, N = x.shape
    # forward reim rfft along N (rfft_half wants (N, batch)); batch = B*Cin
    xr = x.reshape(B * Cin, N).T.astype(dt)                 # (N, B*Cin)
    Yr, Yi = rfft_half(xr, dt)                 # (N//2, B*Cin) real/imag
    Yr = Yr[:m].T.reshape(B, Cin, m).astype(F)             # (B, Cin, m)
    Yi = Yi[:m].T.reshape(B, Cin, m).astype(F)
    # complex channel mix per mode:  (Wr+iWi)(Yr+iYi)
    Wr, Wi = p["w_re"], p["w_im"]
    mix = lambda A, Z: jnp.einsum("iok,bik->bok", A, Z, preferred_element_type=ACC)
    Or_ = mix(Wr, Yr) - mix(Wi, Yi)                        # Re  (B, out_ch, m)
    Oi = mix(Wr, Yi) + mix(Wi, Yr)                         # Im
    # inverse: real field from the m kept modes (reim-style real/imag matmul)
    Cr, Ci = idft_lowmode(m, N, F)                        # (N, m)
    idft = lambda C, Z: jnp.einsum("nk,bok->bon", C, Z, preferred_element_type=ACC)
    return idft(Cr, Or_) - idft(Ci, Oi)                    # (B, out_ch, N) real


#  Layer 2 — linear1x1   (pointwise / 1x1-conv channel mixing: skip, lift, project)
def linear1x1_init(key, in_ch, out_ch):
    std = (2.0 / (in_ch + out_ch)) ** 0.5
    return {"W": std * jax.random.normal(key, (out_ch, in_ch)), "b": jnp.zeros((out_ch,))}


def linear1x1_apply(p, x):
    """x : (B, in_ch, N) -> (B, out_ch, N).  Same weight at every point."""
    return jnp.einsum("oi,bin->bon", p["W"], x) + p["b"][None, :, None]


#  Layer 3 — channel_mlp   (2-layer pointwise MLP; lifting & projection heads)
def channel_mlp_init(key, in_ch, hidden_ch, out_ch):
    k1, k2 = jax.random.split(key)
    return {"l1": linear1x1_init(k1, in_ch, hidden_ch),
            "l2": linear1x1_init(k2, hidden_ch, out_ch)}


def channel_mlp_apply(p, x, act):
    return linear1x1_apply(p["l2"], act(linear1x1_apply(p["l1"], x)))


#  Layer 4 — fno_block   (one Fourier layer:  act(spectral(x) + skip(x)) )
def fno_block_init(key, width, n_modes):
    ks, kl = jax.random.split(key)
    return {"spectral": spectral_conv_init(ks, width, width, n_modes),
            "skip": linear1x1_init(kl, width, width)}


def fno_block_apply(p, x, act):
    return act(spectral_conv_apply(p["spectral"], x) + linear1x1_apply(p["skip"], x))


#  Model — fno   (lifting -> L Fourier blocks -> projection)
def fno_init(key, in_ch, out_ch, width=32, n_modes=16, n_layers=4,
             lift_hidden=None, proj_hidden=None, use_grid=True):
    lift_hidden = lift_hidden or width
    proj_hidden = proj_hidden or width
    lift_in = in_ch + (1 if use_grid else 0)
    keys = jax.random.split(key, n_layers + 2)
    params = {"lifting": channel_mlp_init(keys[0], lift_in, lift_hidden, width),
              "blocks": [fno_block_init(keys[i + 1], width, n_modes) for i in range(n_layers)],
              "projection": channel_mlp_init(keys[-1], width, proj_hidden, out_ch)}
    return params, {"use_grid": use_grid, "n_layers": n_layers}


def fno_apply(params, x, cfg):
    """x : (B, in_ch, N) -> (B, out_ch, N)."""
    if cfg["use_grid"]:
        B, _, N = x.shape
        grid = jnp.broadcast_to(jnp.linspace(0.0, 1.0, N, endpoint=False), (B, 1, N))
        x = jnp.concatenate([x, grid], axis=1)
    x = channel_mlp_apply(params["lifting"], x, jax.nn.gelu)
    for blk in params["blocks"]:
        x = fno_block_apply(blk, x, jax.nn.gelu)
    return channel_mlp_apply(params["projection"], x, jax.nn.gelu)


#  Synthetic operator-learning data:  a(x) -> u = (I - c d^2/dx^2)^{-1} a
#  (periodic Helmholtz smoothing; diagonal in Fourier, so a well-specified FNO fits it).
def make_dataset(key, n_samples, N, k_max=12, c=1e-2):
    ks = jnp.arange(N // 2 + 1)
    key, s1 = jax.random.split(key)
    re = jax.random.normal(s1, (n_samples, N // 2 + 1))
    key, s2 = jax.random.split(key)
    im = jax.random.normal(s2, (n_samples, N // 2 + 1))
    decay = jnp.where(ks <= k_max, 1.0 / (1.0 + ks), 0.0)[None, :]
    a_hat = (re + 1j * im) * decay
    a_hat = a_hat.at[:, 0].set(a_hat[:, 0].real)
    a = jnp.fft.irfft(a_hat, n=N, axis=1)
    g = 1.0 / (1.0 + c * (2.0 * jnp.pi * ks) ** 2)
    u = jnp.fft.irfft(a_hat * g[None, :], n=N, axis=1)
    return a[:, None, :], u[:, None, :]


#  Training  (hand-written Adam over the param pytree; no optax needed)
def mse(pred, y):
    return jnp.mean((pred - y) ** 2)


def rel_l2(pred, y):
    num = jnp.sqrt(jnp.sum((pred - y) ** 2, axis=(-1, -2)))
    den = jnp.sqrt(jnp.sum(y ** 2, axis=(-1, -2)))
    return jnp.mean(num / den)


def adam_init(params):
    z = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
    return {"m": z(params), "v": z(params), "t": 0}


def adam_update(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state["v"], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    new = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                                 params, mhat, vhat)
    return new, {"m": m, "v": v, "t": t}


def train(seed=0, N=256, width=32, n_modes=16, n_layers=4,
          n_train=256, n_test=64, steps=300, batch=32, lr=1e-3):
    key = jax.random.PRNGKey(seed)
    key, kd = jax.random.split(key)
    a, u = make_dataset(kd, n_train + n_test, N)
    xa, ya, xt, yt = a[:n_train], u[:n_train], a[n_train:], u[n_train:]

    key, ki = jax.random.split(key)
    params, cfg = fno_init(ki, in_ch=1, out_ch=1, width=width, n_modes=n_modes, n_layers=n_layers)
    opt = adam_init(params)

    def loss_fn(params, x, y):
        return mse(fno_apply(params, x, cfg), y)

    @jax.jit
    def step(params, opt, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params, opt = adam_update(params, grads, opt, lr=lr)
        return params, opt, loss

    print(f"FNO-1d (reim rfft)  N={N}  width={width}  modes={n_modes}  layers={n_layers}"
          f"  |  {n_train} train / {n_test} test")
    n_batches = max(1, n_train // batch)
    hist = []
    for epoch in range(steps):
        key, ks = jax.random.split(key)
        perm = jax.random.permutation(ks, n_train)
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch:(b + 1) * batch]
            params, opt, loss = step(params, opt, xa[idx], ya[idx])
            ep_loss += float(loss)
        if epoch % 25 == 0 or epoch == steps - 1:
            pred = fno_apply(params, xt, cfg)
            tr, te, rl = ep_loss / n_batches, float(mse(pred, yt)), float(rel_l2(pred, yt))
            hist.append((epoch, tr, te, rl))
            print(f"  epoch {epoch:4d}  train_mse {tr:.3e}   test_mse {te:.3e}   test_relL2 {rl:.3e}")
    return params, cfg, hist


if __name__ == "__main__":
    train()
