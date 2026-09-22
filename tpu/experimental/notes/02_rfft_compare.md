# rfft: Trick A vs Trick B — tested, XLA-inspected, on two engines

> Open in VS Code Markdown preview (`Ctrl/Cmd+Shift+V`) for math. Runnable:
> `python -m tpu.rfft_variants`. Real-input strategies from `01_rfft.md`, measured offline
> (no TPU). Layout throughout: **`(N, K)`** = K independent real signals down the columns.

## What the two are

Both exploit `X[N-k] = conj(X[k])` for real `x`, but on different axes:

- **Trick A (signal pairing).** Pair two real columns `a,b → z = a + i·b`, one **full-length**
  complex FFT, unpack. Halves the **number** of transforms (feeds `(N, K/2)` to the engine).
- **Trick B (even/odd pairing).** Pack `z = x_even + i·x_odd`, one **half-length** complex FFT,
  then a last-layer twist `X[k] = E[k] + W_N^k·O[k]`. Halves the **length** (feeds `(N/2, K)`).

Both return **`(N//2+1, K)`** — the same shape as a plain rfft. (Trick A's `concat(axis=1)`
restores the full K; the `[:N//2+1]` truncates rows via Hermitian symmetry.)

## The code (both, self-contained)

```python
C = jnp.complex64
def _mirror(Z):                       # conj(Z[(M-k) mod M]) along axis 0 — Hermitian partner
    return jnp.conj(jnp.roll(Z[::-1], 1, axis=0))

def rfft_coupled(xr, eng):             # (N, K) real -> (N//2+1, K)
    N, K = xr.shape
    a, b = xr[:, :K//2], xr[:, K//2:]                 # pair signals: a,b are (N, K/2)
    Z = eng((a + 1j*b).astype(C))                     # ONE FFT on K/2 cols -> (N, K/2)
    M = _mirror(Z)
    A  = (0.5 *(Z + M))[:N//2+1]                      # rfft(a): (N//2+1, K/2)
    Bb = (-0.5j*(Z - M))[:N//2+1]                     # rfft(b): (N//2+1, K/2)
    return jnp.concatenate([A, Bb], axis=1)           # (N//2+1, K)

def rfft_reduced(xr, eng):             # (N, K) real -> (N//2+1, K)
    N, K = xr.shape; h = N//2
    z = (xr[0::2] + 1j*xr[1::2]).astype(C)            # (h, K) even + i*odd
    Z = eng(z)                                        # HALF-length FFT -> (h, K)
    M = _mirror(Z)
    E = 0.5 *(Z + M); O = -0.5j*(Z - M)               # even/odd sub-DFTs (length h)
    kf = jnp.arange(h+1); kk = kf % h
    tw = jnp.exp(-2j*jnp.pi*kf/N).astype(C)[:, None]  # recombination twist
    return E[kk] + tw*O[kk]                           # (N//2+1, K)
```

Engines: `jnpfft = jnp.fft.fft` (opaque custom-call, pack/unpack can't fuse in);
`fft_iter` = our radix-B matmul FFT (fusable). Plus `jnp.NATIVE rfft = jnp.fft.rfft`, jnp's
own purpose-built real FFT, as an independent baseline.

## Results (K=64 columns, complex64, XLA static cost, offline)

| N | engine · variant | rel_err | bytes | ×base | flops | ×base |
|---|---|---|---|---|---|---|
| 16384 | **jnp.NATIVE rfft** | 1.6e-07 | **25.2M** | — | 117M | — |
| 16384 | jnpfft · base | 1.6e-07 | 41.9M | 1.00 | 118M | 1.00 |
| 16384 | jnpfft · coupled | 1.7e-07 | 37.8M | 0.90 | 65.0M | 0.55 |
| 16384 | jnpfft · reduced | 1.8e-07 | 37.9M | 0.90 | 62.1M | 0.52 |
| 16384 | fft_iter · base | 1.6e-05 | 76.2M | 1.00 | 540M | 1.00 |
| 16384 | **fft_iter · coupled** | 1.7e-05 | **55.2M** | **0.72** | 276M | **0.51** |
| 16384 | **fft_iter · reduced** | 1.6e-05 | **55.1M** | **0.72** | 210M | **0.39** |

Across N (bytes ×base / flops ×base):

| N | jnpfft A | jnpfft B | fft_iter A | fft_iter B |
|---|---|---|---|---|
| 256 | 0.90 / 0.58 | 0.90 / 0.54 | 0.77 / 0.55 | **0.59** / 0.55 |
| 1024 | 0.90 / 0.57 | 0.90 / 0.53 | 0.74 / 0.53 | 0.74 / 0.52 |
| 16384 | 0.90 / 0.55 | 0.90 / 0.52 | 0.72 / 0.51 | 0.72 / 0.39 |
| 262144 | 0.90 / 0.54 | 0.90 / 0.52 | **0.65** / 0.51 | **0.65** / 0.50 |

## Findings

1. **Flops halve (~0.5×) on both engines** — the Hermitian saving is real; errors stay at the
   dtype floor (fft_iter ~1e-5 f32, jnp ~1e-7). Trick B is marginally the cheapest in flops.
2. **Bytes DROP on both engines** — Trick A/B are a byte win, not a wash:
   - `jnpfft`: **0.90×** (limited — the opaque custom-call can't fuse the pack/unpack, so they
     ride as small extra passes, but the truncated output still nets a win).
   - `fft_iter`: **0.65–0.77×** (bigger — pack/unpack **fuse** into our matmul stages). This is
     the fusion payoff `01_rfft.md` predicted; our own engine cashes it in, the opaque one can't.
3. **`jnp.fft.rfft` (native) is the byte champion** — 25.2M at N=16384, below every variant.
   It's purpose-built C that exploits realness internally. Our `fft_iter`+trick (55M) sits ~2×
   above it — the gap is our matmul-**transpose** overhead (each stage's corner-turn round-trips
   HBM), the same overhead the Pallas VMEM-resident kernel removes (see `03_pallas_compare.md`).
4. **Layout matters — a corrected earlier claim.** An earlier `(K,N)` row-layout run reported
   the tricks *raising* bytes (1.22×) and an interleaved-output scatter hitting 3.11×. In the
   clean `(N, K)` column layout with contiguous `concatenate` (above), that artifact is gone —
   tricks reduce bytes everywhere. Lesson stands: **keep the two unpacked halves contiguous;
   never scatter them back with strided `.at[]`.**

## Verdict

| | Trick A (pair signals) | Trick B (even/odd) |
|---|---|---|
| feeds engine | `(N, K/2)` — half the columns | `(N/2, K)` — half the length |
| needs | ≥2 signals (even K) | single signal ok (K=1) |
| bytes / flops (fft_iter) | 0.65–0.77× / ~0.51× | 0.59–0.77× / ~0.5× |
| FFT engine | unchanged | length changes + twist |

- **On our fusable `fft_iter`, both tricks are a clear win** (bytes ↓ to 0.65–0.77×, flops ↓ ~0.5×).
- **Trick A** suits the FNO (many channels to pair, engine untouched); **Trick B** is the one that
  works for a single signal (K=1).
- **Native `jnp.fft.rfft` still beats us on bytes** — closing that gap is the Pallas job (kill the
  per-stage transpose round-trip), not an rfft-algorithm job.

## Next step

Confirm on a v5e that `fft_iter`+trick's 0.65–0.77× static bytes tracks real wall-clock, and
that the Pallas VMEM-resident rfft (`pallas_rfft`, analytical 0.18×) closes the gap to native.
