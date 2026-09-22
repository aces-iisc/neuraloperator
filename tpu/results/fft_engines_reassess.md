# FFT engine re-assessment (no-complex64), v5e, 2026-09-07

Median blocked wall-clock µs/run, K=256 batch (µs/sample = /256 in parens). us-west1-c, jax 0.11.1.
All engines real/imag — **complex64 op-count = 0** (no c64 conversion; the uniform "1" from the
substring counter is incidental HLO metadata, not a complex op). Ratio 0.5 (half spectrum / low modes).

## 1D rfft — FAIR comparison (part-half computes the SAME N/2 modes as reim/coupled/reduced)
| N | reim | coupled | reduced | part-m16 (16 modes) | part-half (N/2 modes) |
|---|---|---|---|---|---|
| 1,024 | 193.6 | 193.2 | 197.2 | 179.3 | 216.3 |
| 4,096 | **247.8** | 259.6 | 266.8 | 189.7 | 848.7 (3.4× slower) |
| 16,384 | **509.8** | 481.9 | 480.9 | 216.1 | 10,523 (20× slower) |
| 65,536 | 3,779 | **2,217** | 3,108 | 326.6 | 165,343 (44× slower) |
| 262,144 | **17,508** | 19,970 | 18,447 | 883.8 | 2,639,156 (2.6 s, 150× slower) |

**CORRECTED verdict (an earlier draft compared part-m16's 16 modes to the full N/2 — unfair).**
At EQUAL mode count (N/2), the partial is **catastrophically slower** — 150× at 262K (2.6 s vs
17 ms). Partial DFT is O(m·N); at m=N/2 that is O(N²) and explodes. The staged radix FFT is
O(N·log₁₂₈N), near-linear — that is what the FFT is for.
- Partial is the fast path ONLY when m ≪ N (a handful of modes) — the FNO regime (~16 modes).
  There part-m16 stays ~1 pass over x and is fast. It is NOT a faster general FFT.
- For the full/half spectrum, reim/coupled win decisively (by the whole N²/(N log N) factor).
Validated vs jnp.fft.rfft[:m] (rel_err ~1e-6).

## 2D caveat (same lens)
2D `partial` at ratio 0.5 keeps (N/2)² modes vs reim's ~N²/2 — i.e. ~2× fewer modes. So its 3.7×
win at 256² is ~2× mode-count + ~1.85× genuine structure (one pass, no transpose), not a clean 3.7×.

## Staged axis-1 FFT (reim-fast) — TRIED, LOST (do not "fix" the direct axis-1 DFT)
| N | reim (direct axis-1 DFT) | reim-fast (staged axis-1 FFT) | partial |
|---|---|---|---|
| 256² | 4,352 | 5,463 (26% slower) | 1,180 |
| 512² | 19,288 | 22,831 (18% slower) | 7,695 |

`jax_rfft2` does axis-0 as a staged rfft but axis-1 as a DIRECT O(N2²) DFT (a single N2×N2 matmul).
Replacing axis-1 with a staged O(N2 log N2) FFT (`jax_rfft2_fast`, needs 2 transposes) is **18–26%
SLOWER** at large N. Why: TPU is memory-bound + the MXU makes matmul flops ~free, so the direct DFT
= ONE clean matmul pass, while the staged FFT trades free flops for expensive memory movement
(transposes + multi-stage HBM). **The direct axis-1 DFT is already TPU-optimal — not a bug.**
General lesson (same as fused-kernel-loses-to-XLA): brute-force matmul beats clever FFT staging when
memory-bound. The only 2D win is `partial` (direct matmul + fewer modes), and only when few modes suffice.

## Pruned/decimation FFT (rfft_pruned) — BUILT, TESTED at dataset sizes, LOST
Decimate x into r=N/m subseqs -> m-point sub-DFT each -> twiddle-combine (Cooley-Tukey output
pruning; the "compute a smaller size then FFT it" idea). Measured µs/run:
| engine | NS N=128 m=16 K=131072 | Darcy N=32 | Burgers N=16 |
|---|---|---|---|
| partial(m) | **273.4** | 171.7 | 169.3 |
| pruned(m)  | 1422.8 (5.2x slower) | reshape-fail (32%12) | reshape-fail (16%6) |
| reim(full) | 319.8 | 170.7 | 170.7 |
| pallas(full) | 1005.5 | 190.6 | 179.8 |
Verdict: pruned is 5x SLOWER than the direct partial at NS size. Same reason as everything else —
partial is ONE MXU matmul (one HBM pass); pruned's decimate+sub-DFTs+combine is many memory passes,
and memory movement is the bottleneck. At dataset m (<=16) the sub-DFT is too small to stage as an
FFT anyway (pruned == partial in flops, worse in passes). Also needs N % m == 0. Darcy/Burgers are a
flat ~170 us dispatch-floor (transforms too small to differentiate any engine).
CONCLUSION: rfft2_partial (direct low-mode matmul) is the right FFT primitive for the FNO. Confirmed.

## 2D rfft
| N | reim | partial | pallas |
|---|---|---|---|
| 32² | 172.0 | **168.8 (0.66)** | 175.3 |
| 64² | 196.6 | **182.0 (0.71)** | OOM |
| 128² | 342.0 | **293.8 (1.15)** | hang→skip |
| 256² | 4,330.2 (16.9) | **1,161.9 (4.54)** | hang→skip |
| 512² | 19,250.8 (75.2) | **7,668.3 (29.9)** | hang→skip |

## Findings
1. **Small N = flat ~170–200 µs dispatch floor** — engines indistinguishable below ~16K (1D) /
   ~128² (2D). Pallas marginally best only at the tiniest N (one HBM round-trip, whole FFT in VMEM).
2. **Pallas does NOT scale** — it computes the FULL transform in VMEM (not partial), so it OOMs at
   1D ≥4096 and 2D ≥64² (2D ≥128² hangs Mosaic → never run there). Dead end for large N.
3. **2D partial is the decisive winner** — 3.7× at 256², 2.5× at 512², gap widens with N.
   Memory-bound → fewer modes = fewer bytes = faster. (Sharpens the earlier 4.2× one-point result.)
4. **1D coupled has a sweet spot** — 1.7× at 65,536 vs reim; but reim edges back at the extreme top
   (262K). reduced never wins.
5. **No complex64 tax** anywhere — the no-c64 property held for every engine.

Bench: `tpu/experimental/bench_fft_engines.py`.

## HYBRID (full cols + partial rows) — the user's idea, and it WINS
rfft2_hybrid: FULL staged real rfft on axis-0 (all cols), keep low m1; PARTIAL direct DFT on axis-1
(selected low m2). Measured us/run (verified 1e-6 vs rfft2_partial — same modes):
| engine | NS 128^2 m16 K1024 | Darcy 32^2 |
|---|---|---|
| partial (both axes) | 648.7 | 166.6 |
| **hybrid-reim** | **407.7 (1.6x faster)** | 166.3 |
| hybrid-coupled | 910.0 | 171.0 |
| reim (full both) | 1181.3 | 168.4 |
WHY: partial does a SKINNY (m x N)=(16x128) matmul on BOTH axes -> wastes 7/8 of the 128x128 MXU.
The hybrid does axis-0 (the big N2*K=131072-column pass) as a FULL staged rfft (full-tile MXU
efficiency), and keeps the skinny partial only on axis-1 (tiny, over m1*K elements). Full-MXU big
axis + cheap small axis = 1.6x. "Trick or not": NO trick — hybrid-coupled (910) is 2.2x slower than
hybrid-reim (408); packing overhead outweighs the real halving on TPU.
=> Candidate to replace rfft2_partial's axis-0 in the FNO spectral conv for ~1.6x faster FFT.

## Full hybrid sweep (NS 128^2, m16, K1024) — every cols/rows combo, all verified 1e-6
| cols (axis-0) | rows (axis-1) | us | note |
|---|---|---|---|
| reim (staged) | partial | **408.7** | WINNER (1.57x over all-partial) |
| reim | staged-FFT + truncate | 448.7 | "truncated reim rows" — LOSES by 10% |
| partial | partial | 640.4 | old incumbent |
| coupled | partial | 915.5 | |
| reim | full direct DFT | 1198.3 | |
| reduced | partial | 1729.1 | worst (heaviest pack/unpack) |
Darcy 32^2: all ~180-200 us (dispatch floor, indistinguishable).
CONCLUSIONS:
- Big axis (huge N2*K batch): full reim FFT, NO trick (coupled/B both 2.2-4.2x slower — pack/unpack
  is memory-heavy; at N=128 the FFT is 1 MXU-perfect stage so the fixed unpack overhead dominates).
- Small axis (few rows): skinny partial. Staging it (transpose + compute-all-then-slice) loses 10%.
- Optimal low-mode 2D transform = reim cols + partial rows = 1.57x faster than all-partial.

## Hybrid wired into the REAL FNO (end-to-end inference) — TRIED, SLOWER (do not adopt)
spectral_2corner_hybrid (full HxH DFT on axis-0 + select ±band) vs current spectral_2corner (skinny
2m×H partial). End-to-end FNO inference, us/sample (predictions IDENTICAL, rel 0.0):
| dataset | current | hybrid | speedup |
|---|---|---|---|
| NS 128^2 | 237.71 | 267.49 | 0.889x (11% SLOWER) |
| Darcy 32^2 | 22.29 | 23.90 | 0.933x |
| Burgers 17x16 | 4.60 | 4.80 | 0.958x |
WHY the 1.57x FFT microbenchmark REVERSED: the microbench hybrid was SINGLE-corner (staged rfft
half-spectrum, 64 modes, slice 16). The real FNO is 2-CORNER (±band = low +k AND low -k, at opposite
ends of the FULL spectrum), so the hybrid must compute the FULL (128x128) DFT = ALL 128 modes then
select 32 -> a 4x LARGER intermediate (128 vs 32 modes on axis-0). On the memory-bound TPU that extra
HBM traffic erases the full-MXU gain. The skinny partial wastes MXU rows but produces the MINIMAL
intermediate (exactly the ±band) -> fewer bytes wins. CONCLUSION: keep spectral_2corner (skinny
partial) for the 2-corner FNO; the hybrid helps only in the single-corner/half-spectrum microbench.
Lesson: always measure the REAL forward, not the isolated FFT.

## Hermitian 2-corner (compute bottom, top=conj) — TRIED end-to-end, near-wash (not adopted)
spectral_2corner_herm: exploit X[H-k]=conj(X[k]) -> compute only bottom m+1 axis-0 modes, top corner
via free conjugate (half the matmul rows: m+1 vs 2m). Three-way FNO inference us/sample (all identical):
| variant | NS 128^2 | Darcy 32^2 | Burgers |
|---|---|---|---|
| current (skinny 2m×H partial) | 237.6 (best) | 22.4 (best) | 4.55 (best) |
| hermitian | 241.4 (0.984x) | 22.7 (0.988x) | 4.65 (0.975x) |
| full-hybrid | 267.4 (0.889x) | 24.0 | 4.82 |
WHY hermitian is a wash despite halving matmul rows: the 2m×H=(32×128) matmul is already MXU-
underutilized (32<128 rows = ONE tile), so 17 rows is still one tile -> no matmul saving; and the
top-corner conjugate needs a flip(reversal)+concatenate = extra MEMORY ops that cancel it.
FINAL: spectral_2corner (direct skinny partial) is optimal for the 2-corner FNO. Full-MXU loses to a
4x intermediate; fewer-modes loses to flip/concat. No free lunch — current code is at the sweet spot.
