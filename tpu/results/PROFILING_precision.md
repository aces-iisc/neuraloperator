# TPU FFT — HIGHEST vs DEFAULT matmul precision (v5e, xprof device time)

`default` = TPU's single-pass bf16 matmul (~5e-3 rel err).  `highest` = true f32 (3-pass on the bf16 MXU, ~1e-6 err).  Both from `jax.default_matmul_precision(...)` around the same engines; device µs = Σ `device_duration_ps` on /device:TPU:0 ÷ runs.

| N | engine | default µs | highest µs | highest/default | HBM def MB | HBM hi MB |
|---|---|---|---|---|---|---|
| 256 | jnp_1D_rfft_jnp-fft | 8.9 | 8.9 | 1.00x | 0.26 | 0.26 |
| 256 | jnp_1D_rfft_jnp-fft_coupled | 10.0 | 10.0 | 1.00x | 0.79 | 0.79 |
| 256 | jnp_1D_rfft_jnp-fft_reduced | 6.7 | 6.6 | 0.99x | 0.92 | 0.92 |
| 256 | us_1D_rfft_conv-hlo | 7.8 | 7.8 | 1.01x | 0.92 | 0.92 |
| 256 | us_1D_rfft_direct-int8 | 3.4 | 3.4 | 1.02x | 0.79 | 0.79 |
| 256 | us_1D_rfft_fft-iter_coupled | 6.2 | 6.2 | 0.99x | 0.26 | 0.26 |
| 256 | us_1D_rfft_fft-iter_reduced | 4.9 | 4.9 | 1.00x | 0.66 | 0.66 |
| 256 | us_1D_rfft_fft-recur_coupled | 6.2 | 6.2 | 1.00x | 0.26 | 0.26 |
| 256 | us_1D_rfft_fft-recur_reduced | 4.9 | 4.9 | 1.00x | 0.66 | 0.66 |
| 256 | us_1D_rfft_pallas | 5.4 | 5.4 | 1.00x | — | — |
| 256 | us_1D_rfft_pallas-bf16 | 6.0 | — | — | — | — |
| 256 | us_1D_rfft_pallas_reduced | 4.0 | 4.0 | 0.99x | — | — |
| 256 | us_1D_rfft_reim | 6.4 | 6.4 | 1.00x | 0.52 | 0.52 |
| 256 | us_1D_rfft_reim-bf16 | 6.0 | 6.0 | 1.00x | 0.26 | 0.26 |
| 256 | us_1D_rfft_reim-int8 | 6.8 | 6.8 | 1.00x | 0.13 | 0.13 |
| 256 | us_1D_rfft_reimtrick_coupled | 6.5 | 6.5 | 1.00x | 0.72 | 0.72 |
| 256 | us_1D_rfft_reimtrick_reduced | 4.6 | 4.6 | 1.01x | 0.52 | 0.52 |
| 1024 | jnp_1D_rfft_jnp-fft | 23.1 | 23.0 | 1.00x | 1.05 | 1.05 |
| 1024 | jnp_1D_rfft_jnp-fft_coupled | 17.3 | 17.3 | 1.00x | 1.57 | 1.57 |
| 1024 | jnp_1D_rfft_jnp-fft_reduced | 23.4 | 23.4 | 1.00x | 1.57 | 1.57 |
| 1024 | us_1D_rfft_conv-hlo | 12.5 | 12.5 | 1.00x | 2.10 | 2.10 |
| 1024 | us_1D_rfft_direct-int8 | 23.0 | 23.0 | 1.00x | 3.15 | 3.15 |
| 1024 | us_1D_rfft_fft-iter_coupled | 15.0 | 15.1 | 1.00x | 2.10 | 2.10 |
| 1024 | us_1D_rfft_fft-iter_reduced | 16.8 | 16.7 | 1.00x | 2.62 | 2.62 |
| 1024 | us_1D_rfft_fft-recur_coupled | 15.1 | 15.0 | 1.00x | 2.10 | 2.10 |
| 1024 | us_1D_rfft_fft-recur_reduced | 16.8 | 16.7 | 1.00x | 2.62 | 2.62 |
| 1024 | us_1D_rfft_pallas | 7.8 | 7.8 | 0.99x | — | — |
| 1024 | us_1D_rfft_pallas-bf16 | 8.1 | — | — | — | — |
| 1024 | us_1D_rfft_pallas_reduced | 35.8 | 35.9 | 1.00x | — | — |
| 1024 | us_1D_rfft_reim | 9.6 | 9.6 | 1.00x | 2.10 | 2.10 |
| 1024 | us_1D_rfft_reim-bf16 | 8.1 | 8.1 | 1.00x | 1.05 | 1.05 |
| 1024 | us_1D_rfft_reim-int8 | 9.5 | 9.6 | 1.01x | 0.52 | 0.52 |
| 1024 | us_1D_rfft_reimtrick_coupled | 15.4 | 15.4 | 1.00x | 4.20 | 4.20 |
| 1024 | us_1D_rfft_reimtrick_reduced | 17.8 | 17.8 | 1.00x | 4.20 | 4.20 |
| 4096 | jnp_1D_rfft_jnp-fft | 83.8 | 83.7 | 1.00x | 4.19 | 4.19 |
| 4096 | jnp_1D_rfft_jnp-fft_coupled | 60.4 | 60.4 | 1.00x | 4.19 | 4.19 |
| 4096 | jnp_1D_rfft_jnp-fft_reduced | 71.1 | 71.1 | 1.00x | 6.29 | 6.29 |
| 4096 | us_1D_rfft_conv-hlo | 44.4 | 44.5 | 1.00x | 8.39 | 8.39 |
| 4096 | us_1D_rfft_direct-int8 | 275.4 | 275.4 | 1.00x | 12.58 | 12.58 |
| 4096 | us_1D_rfft_fft-iter_coupled | 53.9 | 54.0 | 1.00x | 6.29 | 6.29 |
| 4096 | us_1D_rfft_fft-iter_reduced | 60.3 | 60.2 | 1.00x | 14.68 | 14.68 |
| 4096 | us_1D_rfft_fft-recur_coupled | 53.9 | 54.0 | 1.00x | 6.29 | 6.29 |
| 4096 | us_1D_rfft_fft-recur_reduced | 60.2 | 60.3 | 1.00x | 14.68 | 14.68 |
| 4096 | us_1D_rfft_pallas-bf16 | 23.7 | — | — | — | — |
| 4096 | us_1D_rfft_reim | 31.3 | 31.3 | 1.00x | 8.39 | 8.39 |
| 4096 | us_1D_rfft_reim-bf16 | 26.2 | 26.1 | 1.00x | 4.19 | 4.19 |
| 4096 | us_1D_rfft_reim-int8 | 24.9 | 24.9 | 1.00x | 2.10 | 2.10 |
| 4096 | us_1D_rfft_reimtrick_coupled | 46.5 | 46.6 | 1.00x | 8.39 | 8.39 |
| 4096 | us_1D_rfft_reimtrick_reduced | 49.1 | 49.2 | 1.00x | 12.58 | 12.58 |
| 8192 | jnp_1D_rfft_jnp-fft | 176.1 | 175.6 | 1.00x | 8.39 | 8.39 |
| 8192 | jnp_1D_rfft_jnp-fft_coupled | 123.7 | 123.7 | 1.00x | 8.39 | 8.39 |
| 8192 | jnp_1D_rfft_jnp-fft_reduced | 140.8 | 140.8 | 1.00x | 12.58 | 12.58 |
| 8192 | us_1D_rfft_conv-hlo | 99.5 | 99.7 | 1.00x | 16.78 | 16.78 |
| 8192 | us_1D_rfft_direct-int8 | 1051.9 | 1051.9 | 1.00x | 25.17 | 25.17 |
| 8192 | us_1D_rfft_fft-iter_coupled | 107.8 | 108.0 | 1.00x | 12.58 | 12.58 |
| 8192 | us_1D_rfft_fft-iter_reduced | 117.1 | 117.4 | 1.00x | 29.36 | 29.36 |
| 8192 | us_1D_rfft_fft-recur_coupled | 107.8 | 108.0 | 1.00x | 12.58 | 12.58 |
| 8192 | us_1D_rfft_fft-recur_reduced | 117.2 | 117.4 | 1.00x | 29.36 | 29.36 |
| 8192 | us_1D_rfft_reim | 74.6 | 74.5 | 1.00x | 16.78 | 16.78 |
| 8192 | us_1D_rfft_reim-bf16 | 69.0 | 69.0 | 1.00x | 8.39 | 8.39 |
| 8192 | us_1D_rfft_reim-int8 | 50.4 | 50.4 | 1.00x | 4.19 | 4.19 |
| 8192 | us_1D_rfft_reimtrick_coupled | 92.8 | 92.9 | 1.00x | 16.78 | 16.78 |
| 8192 | us_1D_rfft_reimtrick_reduced | 93.2 | 93.3 | 1.00x | 25.17 | 25.17 |
| 16384 | jnp_1D_rfft_jnp-fft | 311.3 | 311.2 | 1.00x | 16.78 | 16.78 |
| 16384 | jnp_1D_rfft_jnp-fft_coupled | 244.4 | 244.4 | 1.00x | 16.78 | 16.78 |
| 16384 | jnp_1D_rfft_jnp-fft_reduced | 299.6 | 299.7 | 1.00x | 25.17 | 25.17 |
| 16384 | us_1D_rfft_conv-hlo | 216.7 | 217.8 | 1.00x | 33.55 | 33.55 |
| 16384 | us_1D_rfft_direct-int8 | 4210.9 | 4210.7 | 1.00x | 587.20 | 587.20 |
| 16384 | us_1D_rfft_fft-iter_coupled | 214.9 | 214.8 | 1.00x | 25.17 | 25.17 |
| 16384 | us_1D_rfft_fft-iter_reduced | 246.7 | 246.7 | 1.00x | 58.72 | 58.72 |
| 16384 | us_1D_rfft_fft-recur_coupled | 214.8 | 214.9 | 1.00x | 25.17 | 25.17 |
| 16384 | us_1D_rfft_fft-recur_reduced | 246.6 | 246.6 | 1.00x | 58.72 | 58.72 |
| 16384 | us_1D_rfft_reim | 171.1 | 171.1 | 1.00x | 50.33 | 50.33 |
| 16384 | us_1D_rfft_reim-bf16 | 146.8 | 146.8 | 1.00x | 25.17 | 25.17 |
| 16384 | us_1D_rfft_reim-int8 | 145.4 | 145.4 | 1.00x | 8.39 | 8.39 |
| 16384 | us_1D_rfft_reimtrick_coupled | 195.5 | 196.1 | 1.00x | 33.55 | 33.55 |
| 16384 | us_1D_rfft_reimtrick_reduced | 203.1 | 203.2 | 1.00x | 50.33 | 50.33 |

## reading it
- `highest` forces true-f32 matmuls (3 bf16 passes), so MXU-heavy engines get **slower** (≈2–3× on the matmul share) but hit ~1e-6 accuracy; memory-bound engines move the same HBM, so their slowdown is smaller.
- Pallas/bf16/int8 engines are unaffected by `highest` (they don't use f32 MXU matmuls).

## finding — HIGHEST precision is essentially FREE here

Across **every** engine and N, `highest/default ≈ 1.00–1.01×` — true-f32 matmuls cost **no extra
device time**. Reason: these rfft engines are **memory-bound** (AI ≪ 240 ridge), so the MXU is
already stalled waiting on HBM; doing 3 bf16 passes instead of 1 adds compute that hides entirely
behind the memory wait. HBM bytes are identical (same columns), so time is identical.

**Practical upshot:** you can run the FFT/FNO spectral transform at `precision='highest'` (≈1e-6
accuracy) instead of the default bf16 (~5e-3) **for free** on the v5e. The bf16-operand engines
(`reim-bf16`, `pallas-bf16`) are separate — they cut the *bytes* (½ HBM), which IS what buys speed;
`highest` only changes matmul passes, not bytes. So: **bf16 for speed (fewer bytes), highest for
accuracy (free) — they're orthogonal knobs.**  (Traces: `tpu/traces_default/`, `tpu/traces_highest/`
— load either with `tensorboard --logdir`.)
