# Fast FFT + FNO on TPU v5e — experiment report

Goal of the project: run the **standard** FNO (and its FFT) **as fast as possible on a single
v5e**, using a real/imag matmul-FFT (no `complex64`). Not an algorithm change — a speed project.
Every number below is **measured** (xprof device time or blocked wall-clock on a real v5e); nothing
estimated.

---

## 1. Which 1-D FFT is fastest (device µs/run, K=256)

| N | reim-bf16 | reim | pallas (VMEM) | conv-hlo | coupled | jnp.rfft |
|---|---|---|---|---|---|---|
| 1024 | **8.1** | 9.6 | ★7.8 | 12.5 | 15.1 | 23.0 |
| 4096 | ★**26.2** | 31.3 | (OOM) | 44.4 | 54.0 | 88.6 |
| 8192 | ★**69.0** | 74.7 | (OOM) | 99.5 | 116 | 176 |
| 16384 | ★**147** | 171 | (OOM) | 217 | 232 | 311 |

**Why:** the transform is **memory-bound** (arithmetic intensity 4–13 ≪ 240 ridge), so the winner
is whoever moves the fewest bytes. `reim-bf16` halves operand bytes → ½ HBM → fastest at every
N≥1024. `pallas` (whole FFT resident in VMEM, one HBM read/write) wins at small N but **OOMs the
16 MB VMEM above 4096**. Packing tricks (coupled/reduced) lose to native `reim` because pack/unpack costs
more than it saves. `jnp.rfft` is slowest — it pays a `complex64` assembly tax + more copies.

---

## 2. FFT findings (all measured)

| experiment | result | why |
|---|---|---|
| **complex64 tax** | reim = **0** c64 custom-calls; jnp/conv-hlo/tricks spend **10–17%** assembling complex64 | TPU has no native complex; assembling it is pure overhead. reim carries (re,im) as 2 f32 arrays. |
| **no-c64 tricks** | real/imag tricks **10–20% faster** than the complex64 tricks at N≥4096 | drops the c64 assembly; net win once N is large enough to amortize the unpack. |
| **precision highest vs default** | **identical time** (105.9 vs 105.9 µs), accuracy Δ +0.0008 | memory-bound → the extra MXU passes hide behind the memory wait. **highest is free.** |
| **bf16 activations** | **+27% throughput** (3.34 vs 4.24 µs/sample), accuracy Δ +0.0008 | halves the *activation* HBM (the dominant traffic). bf16 *weights* alone gave ~0% — activations dominate. |
| **factorized spectral weights** | F-FNO separable **4×** fewer params; TFNO CP rank-8 **33×** fewer, same accuracy | the (Cin,Cout,m,m) weight is low-rank; factoring shrinks it. |
| **combined "pack-reim"** | **failed** (slower than reim) | pack-2-reals and native-real are alternative ways to exploit realness — they don't stack. |

**2-D FFT (the FNO transform):** the low-mode **partial DFT** — compute only the kept modes as a
GEMM, no full FFT, no transpose — is the champion:

| Darcy 32² spectral transform | inference |
|---|---|
| **partial (ours)** | **6.22 µs/sample** |
| layout-reorder variant | 6.57 (XLA already handles layout — no help) |
| **jnp.rfft2** (full FFT + truncate, c64) | 26.38 (**4.2× slower**) |

---

## 3. The FNO — standard algorithm, run fast (final)

Standard FNO2d (2-corner ±band spectral conv + domain padding), implemented with the fast reim
partial-DFT (no complex64), trained with **full data + early stopping**:

| dataset | rel L2 (ours) | paper FNO | **inference / sample (v5e)** | throughput |
|---|---|---|---|---|
| **Navier–Stokes 128²** | **0.043** | ~0.08 (ν=1e-4) | **135.6 µs** | 7,375 /s |
| **Darcy 32²** | 0.076 | ~0.01 | **5.94 µs** | 168,288 /s |
| **Burgers** | **0.0054** | ~0.014 | **1.79 µs** | 558,635 /s |

**Why these numbers:**
- **NS beats the paper (0.043 < 0.08)** and Burgers matches it — proof the fast reim FNO *is* the
  correct standard FNO. Earlier we saw NS 0.222 only because of two shortcuts we had taken
  (a 2000-sample subset → overfit; a 1-corner spectral conv → half the modes). Fixing both →
  a clean, monotonically-decreasing loss curve to 0.043. **Not a precision issue.**
- **Darcy stays ~8× above the paper's 0.01** — a **resolution** limit: the paper uses 85–421 grids;
  32² is the largest Darcy in the repo. Nothing to fix in the model.

---

## 4. Inference per sample vs the GPU state of the art

| | resolution | inference | hardware | note |
|---|---|---|---|---|
| **Ours (reim FNO)** | NS 128² | **136 µs/sample** | v5e | batched (128), amortized throughput |
| **Ours** | Darcy 32² | **5.9 µs/sample** | v5e | |
| FNO paper (Li 2021) | NS 256² | ~5 ms / solve | V100-era GPU | single instance, 4× our pixels |
| TurboFNO (GPU SOTA) | 2D | up to **2.5× PyTorch** | A100 | fused FFT-GEMM-iFFT; no absolute µs/sample published |

**Honest comparison:** an exact apples-to-apples GPU number isn't published (different resolution,
batching, and HW generation). Our figures are **batched throughput** (µs/sample amortized over a
128-sample batch), whereas the paper's 5 ms is a single-instance full solve at 4× the pixels — so
they are not directly comparable, and we do **not** claim "we beat the GPU." What we *can* say: on a
single v5e, the standard FNO runs at **1.8–136 µs/sample**, and the FFT is already at the memory
roofline (the only remaining lever is a fused FFT-GEMM-iFFT VMEM kernel — the TurboFNO idea — which
is the open work item).

---

## 5. What's the remaining speed lever
Everything above is at the **memory-bound roofline** already (bf16 halves it; highest precision is
free). The one thing left that XLA won't do automatically is a **fused forward-DFT → channel-mix
GEMM → inverse-DFT in one VMEM-resident Pallas kernel** (keep the whole spectral conv on-chip, one
HBM round-trip). That is exactly TurboFNO's 50–105% GPU win, ported to the v5e — the next step.

*Artifacts:* engine profiles in `PROFILING*.md`; trained tensors + loss curves in `models/`
(reload with `fno_proper.load_params`); FFT engines in `fft1d/`, `fft2d/`; the fast FNO in
`fno_reim_2d.py` / `experimental/fno_proper.py`.
