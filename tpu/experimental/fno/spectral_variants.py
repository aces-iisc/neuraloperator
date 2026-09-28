import jax.numpy as jnp

from ...fft.core import ACC, F
from ...fno.spectral import _dft_rows as _rows

_E = lambda spec, mat, x: jnp.einsum(spec, mat, x, preferred_element_type=ACC)


def spectral_2corner_hybrid(w, x, n_modes):
    """Same 2-corner spectral conv as spectral_2corner, but axis-0 uses the FULL (H×H) DFT (a full
    128×128 MXU matmul at 100% utilization) and then SELECTS the ±m band — instead of the skinny
    (2m×H) partial matmul that wastes most of the MXU.  axis-0 is the big-batch axis (n·c·W columns),
    so full-MXU efficiency there wins (the measured 'reim cols + partial rows' hybrid).  Identical
    output to spectral_2corner (same modes).  axis-1 stays partial; inverse unchanged."""
    m = n_modes // 2        # n_modes counts both corners
    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    k1 = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])   # ± band, 2m indices

    # forward axis-0: FULL (H×H) DFT (full MXU), then select the ±band  <-- the only change
    Cr0f, Ci0f = _rows(jnp.arange(H), H, -1)  # (H, H) full DFT
    Arf = _E("ah,nchw->ncaw", Cr0f, x)  # (n,c,H,W) all modes
    Aif = _E("ah,nchw->ncaw", Ci0f, x)

    Ar = jnp.concatenate([Arf[:, :, :m], Arf[:, :, H - m:]], axis=2)   # ±band -> (n,c,2m,W)
    Ai = jnp.concatenate([Aif[:, :, :m], Aif[:, :, H - m:]], axis=2)

    # forward axis-1 (partial, low m) — identical to spectral_2corner
    Cr1, Ci1 = _rows(jnp.arange(m), W, -1)
    ew = lambda M, Z: _E("dw,ncaw->ncad", M, Z)
    Xr = ew(Cr1, Ar) - ew(Ci1, Ai); Xi = ew(Cr1, Ai) + ew(Ci1, Ar)
    mix = lambda M, Z: _E("ioad,niad->noad", M, Z)

    Or_ = mix(w["w_re"], Xr) - mix(w["w_im"], Xi); Oi = mix(w["w_re"], Xi) + mix(w["w_im"], Xr)
    d = jnp.arange(m); alpha = jnp.where(d == 0, 1.0, 2.0).astype(F)
    IwR, IwI = _rows(d, W, +1)
    IwR, IwI = (IwR * alpha[:, None]).T, (IwI * alpha[:, None]).T
    iw = lambda M, Z: _E("wd,noad->noaw", M, Z)

    Pr = iw(IwR, Or_) - iw(IwI, Oi); Pi = iw(IwR, Oi) + iw(IwI, Or_)
    IhR, IhI = _rows(k1, H, +1)
    ih = lambda M, Z: _E("ha,noaw->nohw", M.T, Z)
    u = ih(IhR, Pr) - ih(IhI, Pi)
    
    return (u / (H * W)).astype(F)


def spectral_2corner_herm(w, x, n_modes):
    """Same 2-corner conv, but exploit HERMITIAN symmetry of the real axis-0 transform:
    X[H-k] = conj(X[k]).  So compute ONLY the low-positive modes X[0..m] (m+1 rows) with a
    ((m+1)×H) DFT, then get the NEGATIVE (top) corner FREE as the conjugate — halving the axis-0
    forward matmul rows (m+1 vs 2m).  Same modes/output as spectral_2corner."""
    m = n_modes // 2        # n_modes counts both corners

    H, W = x.shape[-2], x.shape[-1]
    x = x.astype(F)
    k1 = jnp.concatenate([jnp.arange(m), jnp.arange(H - m, H)])

    # forward axis-0: ONLY the low-positive modes [0..m] (m+1 rows)  <-- half the current 2m rows
    Cr0, Ci0 = _rows(jnp.arange(m + 1), H, -1)                   # (m+1, H)
    Ap_r = _E("ah,nchw->ncaw", Cr0, x); Ap_i = _E("ah,nchw->ncaw", Ci0, x)   # (n,c,m+1,W)

    # ±band (k1 order): bottom = X[0..m); top = conj(X[m],X[m-1],...,X[1])  (free conjugate)
    Ar = jnp.concatenate([Ap_r[:, :, :m], Ap_r[:, :, m:0:-1]], axis=2)       # (n,c,2m,W)
    Ai = jnp.concatenate([Ap_i[:, :, :m], -Ap_i[:, :, m:0:-1]], axis=2)      # top imag negated

    # identical from here (axis-1 partial, mix, inverse)
    Cr1, Ci1 = _rows(jnp.arange(m), W, -1)
    ew = lambda M, Z: _E("dw,ncaw->ncad", M, Z)
    Xr = ew(Cr1, Ar) - ew(Ci1, Ai); Xi = ew(Cr1, Ai) + ew(Ci1, Ar)

    mix = lambda M, Z: _E("ioad,niad->noad", M, Z)
    Or_ = mix(w["w_re"], Xr) - mix(w["w_im"], Xi); Oi = mix(w["w_re"], Xi) + mix(w["w_im"], Xr)
    d = jnp.arange(m); alpha = jnp.where(d == 0, 1.0, 2.0).astype(F)
    IwR, IwI = _rows(d, W, +1)
    IwR, IwI = (IwR * alpha[:, None]).T, (IwI * alpha[:, None]).T
    iw = lambda M, Z: _E("wd,noad->noaw", M, Z)

    Pr = iw(IwR, Or_) - iw(IwI, Oi); Pi = iw(IwR, Oi) + iw(IwI, Or_)
    IhR, IhI = _rows(k1, H, +1)
    ih = lambda M, Z: _E("ha,noaw->nohw", M.T, Z)
    u = ih(IhR, Pr) - ih(IhI, Pi)
    return (u / (H * W)).astype(F)


def spectral_2corner_jnpfft(w, x, n_modes):
    m = n_modes // 2        # n_modes counts both corners
    B, C, H, W = x.shape
    Xf = jnp.fft.rfft2(x, axes=(-2, -1))
    Wc = w["w_re"] + 1j * w["w_im"]
    out = jnp.zeros((B, w["w_re"].shape[1], H, W // 2 + 1), Xf.dtype)
    top = jnp.einsum("ioad,niad->noad", Wc[:, :, :m], Xf[..., :m, :m])
    bot = jnp.einsum("ioad,niad->noad", Wc[:, :, m:], Xf[..., H - m:, :m])
    out = out.at[..., :m, :m].set(top).at[..., H - m:, :m].set(bot)
    return jnp.fft.irfft2(out, s=(H, W), axes=(-2, -1))
