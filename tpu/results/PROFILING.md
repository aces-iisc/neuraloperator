# TPU FFT profiling — tpu/traces/ (xprof /device:TPU:0)

## methodology
- capture: `bench_jax.profile()` = `jax.profiler.trace()` around 15 `block_until_ready(fn(x))` per (engine, N).
- device µs = Σ `XLA Ops` event `device_duration_ps` on /device:TPU:0 ÷ runs (on-chip time only, NO host dispatch).
- HBM = `memory_access_breakdown` mem-space-1 bytes ÷ runs.  AI = `flops` ÷ `bytes_accessed`.  category = XLA `hlo_category`.

## verify in TensorBoard (open this trace; each number's source)
| number here | TensorBoard tool → field |
|---|---|
| device µs/run | Trace Viewer → 'XLA Modules' block duration; or Op Profile → total self-time ÷ runs |
| HBM MB/run | Memory Viewer → peak/bytes; or Op Profile → 'Bytes accessed' (HBM), ÷ runs |
| AI (flop/byte) | Op Profile → 'FLOPs' ÷ 'Bytes accessed' |
| MXU/VPU/memops/custom % | Op Profile → group by Category (rolled up from hlo_category) |
| the RAW category rows below | Op Profile → the 'Category' column, ungrouped |
| a single op (fusion.N) | Op Profile → expand a category; or Trace Viewer → click the op |

## summary  (sorted by N;  ★ = fastest device µs at that N;  the engine name is split into who/D/func/method/trick facets;  memops = copy+reshape+transpose+slice+gather+reverse+DMA)

| N | engine | who | D | func | method | trick | device µs | HBM MB | AI | MXU% | VPU% | memops% | custom% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 256 | ★ us_1D_rfft_direct-int8 | us | 1D | rfft | direct-int8 | — | 3.4 | 0.79 | 8 | 32 | 68 | 0 | 0 |
| 256 | us_1D_rfft_pallas_reduced | us | 1D | rfft | pallas | reduced | 4.0 | — | 0 | 0 | 0 | 0 | 100 |
| 256 | us_1D_rfft_fft-recur_reduced | us | 1D | rfft | fft-recur | reduced | 4.9 | 0.66 | 5 | 38 | 8 | 38 | 17 |
| 256 | us_1D_rfft_fft-iter_reduced | us | 1D | rfft | fft-iter | reduced | 4.9 | 0.66 | 5 | 38 | 8 | 38 | 17 |
| 256 | us_1D_rfft_pallas | us | 1D | rfft | pallas | — | 5.4 | — | 0 | 0 | 0 | 0 | 100 |
| 256 | us_1D_fft_pallas | us | 1D | fft | pallas | — | 5.6 | — | 0 | 0 | 0 | 0 | 100 |
| 256 | us_1D_rfft_pallas-bf16 | us | 1D | rfft | pallas-bf16 | — | 6.0 | — | 0 | 0 | 0 | 0 | 100 |
| 256 | us_1D_rfft_reim-bf16 | us | 1D | rfft | reim-bf16 | — | 6.0 | 0.26 | 9 | 44 | 19 | 37 | 0 |
| 256 | us_1D_rfft_fft-recur_coupled | us | 1D | rfft | fft-recur | coupled | 6.2 | 0.26 | 4 | 48 | 17 | 22 | 13 |
| 256 | us_1D_rfft_fft-iter_coupled | us | 1D | rfft | fft-iter | coupled | 6.2 | 0.26 | 4 | 47 | 18 | 22 | 13 |
| 256 | us_1D_rfft_reim | us | 1D | rfft | reim | — | 6.4 | 0.52 | 6 | 41 | 17 | 41 | 0 |
| 256 | jnp_1D_rfft_jnp-fft_reduced | jnp | 1D | rfft | jnp-fft | reduced | 6.6 | 0.92 | 24 | 39 | 14 | 35 | 12 |
| 256 | us_1D_rfft_reim-int8 | us | 1D | rfft | reim-int8 | — | 6.8 | 0.13 | 6 | 27 | 41 | 31 | 0 |
| 256 | us_1D_fft_fft-recur | us | 1D | fft | fft-recur | — | 7.4 | 0.26 | 7 | 46 | 2 | 36 | 16 |
| 256 | us_1D_fft_fft-iter | us | 1D | fft | fft-iter | — | 7.5 | 0.26 | 7 | 46 | 2 | 36 | 15 |
| 256 | us_1D_rfft_conv-hlo | us | 1D | rfft | conv-hlo | — | 7.8 | 0.92 | 5 | 53 | 6 | 24 | 11 |
| 256 | jnp_1D_rfft_jnp-fft | jnp | 1D | rfft | jnp-fft | — | 8.9 | 0.26 | 154 | 67 | 24 | 0 | 9 |
| 256 | jnp_1D_fft_jnp-fft | jnp | 1D | fft | jnp-fft | — | 9.0 | 0.26 | 154 | 65 | 22 | 0 | 13 |
| 256 | jnp_1D_rfft_jnp-fft_coupled | jnp | 1D | rfft | jnp-fft | coupled | 10.0 | 0.79 | 50 | 82 | 1 | 9 | 8 |
| 1024 | ★ us_1D_rfft_pallas | us | 1D | rfft | pallas | — | 7.8 | — | 0 | 0 | 0 | 0 | 100 |
| 1024 | us_1D_rfft_reim-bf16 | us | 1D | rfft | reim-bf16 | — | 8.1 | 1.05 | 9 | 47 | 6 | 47 | 0 |
| 1024 | us_1D_rfft_pallas-bf16 | us | 1D | rfft | pallas-bf16 | — | 8.1 | — | 0 | 0 | 0 | 0 | 100 |
| 1024 | us_1D_fft_pallas | us | 1D | fft | pallas | — | 8.8 | — | 0 | 0 | 0 | 0 | 100 |
| 1024 | us_1D_rfft_reim-int8 | us | 1D | rfft | reim-int8 | — | 9.5 | 0.52 | 7 | 30 | 23 | 46 | 0 |
| 1024 | us_1D_rfft_reim | us | 1D | rfft | reim | — | 9.6 | 2.10 | 6 | 41 | 5 | 53 | 0 |
| 1024 | us_1D_rfft_conv-hlo | us | 1D | rfft | conv-hlo | — | 12.4 | 2.10 | 7 | 60 | 6 | 17 | 15 |
| 1024 | us_1D_fft_fft-recur | us | 1D | fft | fft-recur | — | 15.0 | 2.10 | 7 | 47 | 1 | 29 | 22 |
| 1024 | us_1D_fft_fft-iter | us | 1D | fft | fft-iter | — | 15.0 | 2.10 | 7 | 47 | 1 | 29 | 22 |
| 1024 | us_1D_rfft_fft-recur_coupled | us | 1D | rfft | fft-recur | coupled | 15.1 | 2.10 | 3 | 36 | 18 | 34 | 12 |
| 1024 | us_1D_rfft_fft-iter_coupled | us | 1D | rfft | fft-iter | coupled | 15.1 | 2.10 | 3 | 35 | 18 | 34 | 12 |
| 1024 | us_1D_rfft_fft-recur_reduced | us | 1D | rfft | fft-recur | reduced | 16.8 | 2.62 | 3 | 25 | 7 | 53 | 11 |
| 1024 | us_1D_rfft_fft-iter_reduced | us | 1D | rfft | fft-iter | reduced | 16.8 | 2.62 | 3 | 25 | 7 | 52 | 11 |
| 1024 | jnp_1D_rfft_jnp-fft_coupled | jnp | 1D | rfft | jnp-fft | coupled | 17.3 | 1.57 | 22 | 56 | 11 | 22 | 11 |
| 1024 | us_1D_rfft_direct-int8 | us | 1D | rfft | direct-int8 | — | 23.0 | 3.15 | 27 | 15 | 84 | 1 | 0 |
| 1024 | jnp_1D_rfft_jnp-fft | jnp | 1D | rfft | jnp-fft | — | 23.0 | 1.05 | 47 | 74 | 1 | 17 | 8 |
| 1024 | jnp_1D_rfft_jnp-fft_reduced | jnp | 1D | rfft | jnp-fft | reduced | 23.4 | 1.57 | 21 | 54 | 6 | 29 | 8 |
| 1024 | jnp_1D_fft_jnp-fft | jnp | 1D | fft | jnp-fft | — | 24.3 | 1.05 | 51 | 70 | 1 | 16 | 13 |
| 1024 | us_1D_rfft_pallas_reduced | us | 1D | rfft | pallas | reduced | 35.9 | — | 0 | 0 | 0 | 0 | 100 |
| 64x64 | ★ us_2D_rfft_partial | us | 2D | rfft | partial | — | 16.7 | 6.29 | 8 | 53 | 0 | 47 | 0 |
| 64x64 | us_2D_rfft_reim | us | 2D | rfft | reim | — | 21.0 | 8.39 | 7 | 52 | 2 | 45 | 0 |
| 64x64 | jnp_2D_rfft_jnp-fft | jnp | 2D | rfft | jnp-fft | — | 51.1 | 4.19 | 54 | 83 | 4 | 0 | 13 |
| 64x64 | jnp_2D_fft_jnp-fft | jnp | 2D | fft | jnp-fft | — | 83.6 | 8.39 | 55 | 85 | 0 | 0 | 15 |
| 4096 | ★ us_1D_rfft_pallas-bf16 | us | 1D | rfft | pallas-bf16 | — | 23.7 | — | 0 | 0 | 0 | 0 | 100 |
| 4096 | us_1D_rfft_reim-int8 | us | 1D | rfft | reim-int8 | — | 24.9 | 2.10 | 8 | 44 | 14 | 42 | 0 |
| 4096 | us_1D_rfft_reim-bf16 | us | 1D | rfft | reim-bf16 | — | 26.2 | 4.19 | 11 | 56 | 3 | 42 | 0 |
| 4096 | us_1D_rfft_reim | us | 1D | rfft | reim | — | 31.3 | 8.39 | 7 | 47 | 3 | 50 | 0 |
| 4096 | us_1D_rfft_conv-hlo | us | 1D | rfft | conv-hlo | — | 44.3 | 8.39 | 8 | 59 | 7 | 16 | 15 |
| 4096 | us_1D_rfft_fft-recur_coupled | us | 1D | rfft | fft-recur | coupled | 53.9 | 6.29 | 4 | 41 | 16 | 31 | 12 |
| 4096 | us_1D_rfft_fft-iter_coupled | us | 1D | rfft | fft-iter | coupled | 54.0 | 6.29 | 4 | 41 | 16 | 31 | 12 |
| 4096 | us_1D_fft_fft-recur | us | 1D | fft | fft-recur | — | 57.2 | 8.39 | 8 | 43 | 1 | 34 | 22 |
| 4096 | us_1D_fft_fft-iter | us | 1D | fft | fft-iter | — | 57.3 | 8.39 | 8 | 43 | 1 | 34 | 23 |
| 4096 | us_1D_rfft_fft-recur_reduced | us | 1D | rfft | fft-recur | reduced | 60.3 | 14.68 | 3 | 22 | 7 | 58 | 11 |
| 4096 | us_1D_rfft_fft-iter_reduced | us | 1D | rfft | fft-iter | reduced | 60.3 | 14.68 | 3 | 22 | 7 | 58 | 11 |
| 4096 | jnp_1D_rfft_jnp-fft_coupled | jnp | 1D | rfft | jnp-fft | coupled | 60.4 | 4.19 | 29 | 57 | 16 | 16 | 11 |
| 4096 | jnp_1D_rfft_jnp-fft_reduced | jnp | 1D | rfft | jnp-fft | reduced | 71.1 | 6.29 | 24 | 49 | 7 | 35 | 9 |
| 4096 | jnp_1D_rfft_jnp-fft | jnp | 1D | rfft | jnp-fft | — | 83.7 | 4.19 | 55 | 73 | 1 | 18 | 8 |
| 4096 | jnp_1D_fft_jnp-fft | jnp | 1D | fft | jnp-fft | — | 94.4 | 4.19 | 60 | 71 | 0 | 15 | 14 |
| 4096 | us_1D_rfft_direct-int8 | us | 1D | rfft | direct-int8 | — | 275.5 | 12.58 | 90 | 9 | 91 | 0 | 0 |
| 8192 | ★ us_1D_rfft_reim-int8 | us | 1D | rfft | reim-int8 | — | 50.4 | 4.19 | 10 | 44 | 12 | 44 | 0 |
| 8192 | us_1D_rfft_reim-bf16 | us | 1D | rfft | reim-bf16 | — | 69.0 | 8.39 | 14 | 39 | 2 | 60 | 0 |
| 8192 | us_1D_rfft_reim | us | 1D | rfft | reim | — | 74.6 | 16.78 | 9 | 39 | 3 | 58 | 0 |
| 8192 | us_1D_rfft_conv-hlo | us | 1D | rfft | conv-hlo | — | 99.5 | 16.78 | 9 | 48 | 6 | 34 | 13 |
| 8192 | us_1D_rfft_fft-recur_coupled | us | 1D | rfft | fft-recur | coupled | 107.8 | 12.58 | 5 | 40 | 17 | 31 | 12 |
| 8192 | us_1D_rfft_fft-iter_coupled | us | 1D | rfft | fft-iter | coupled | 107.9 | 12.58 | 5 | 40 | 17 | 31 | 12 |
| 8192 | us_1D_rfft_fft-recur_reduced | us | 1D | rfft | fft-recur | reduced | 117.1 | 29.36 | 3 | 21 | 7 | 58 | 11 |
| 8192 | us_1D_rfft_fft-iter_reduced | us | 1D | rfft | fft-iter | reduced | 117.2 | 29.36 | 3 | 21 | 7 | 58 | 11 |
| 8192 | jnp_1D_rfft_jnp-fft_coupled | jnp | 1D | rfft | jnp-fft | coupled | 123.7 | 8.39 | 35 | 57 | 16 | 17 | 10 |
| 8192 | us_1D_fft_fft-recur | us | 1D | fft | fft-recur | — | 129.3 | 16.78 | 9 | 35 | 0 | 45 | 20 |
| 8192 | us_1D_fft_fft-iter | us | 1D | fft | fft-iter | — | 129.3 | 16.78 | 9 | 35 | 0 | 45 | 20 |
| 8192 | jnp_1D_rfft_jnp-fft_reduced | jnp | 1D | rfft | jnp-fft | reduced | 140.8 | 12.58 | 26 | 48 | 6 | 36 | 9 |
| 8192 | jnp_1D_rfft_jnp-fft | jnp | 1D | rfft | jnp-fft | — | 176.1 | 8.39 | 67 | 74 | 1 | 17 | 7 |
| 8192 | jnp_1D_fft_jnp-fft | jnp | 1D | fft | jnp-fft | — | 187.4 | 8.39 | 72 | 70 | 0 | 16 | 14 |
| 8192 | us_1D_rfft_direct-int8 | us | 1D | rfft | direct-int8 | — | 1051.8 | 25.17 | 89 | 9 | 91 | 1 | 0 |
| 128x128 | ★ us_2D_rfft_partial | us | 2D | rfft | partial | — | 65.0 | 25.17 | 15 | 46 | 0 | 54 | 0 |
| 128x128 | us_2D_rfft_reim | us | 2D | rfft | reim | — | 98.6 | 33.55 | 11 | 17 | 1 | 82 | 0 |
| 128x128 | jnp_2D_rfft_jnp-fft | jnp | 2D | rfft | jnp-fft | — | 228.6 | 33.82 | 108 | 72 | 2 | 24 | 1 |
| 128x128 | jnp_2D_fft_jnp-fft | jnp | 2D | fft | jnp-fft | — | 310.3 | 50.33 | 105 | 72 | 0 | 26 | 2 |
| 16384 | ★ us_1D_rfft_reim-int8 | us | 1D | rfft | reim-int8 | — | 145.4 | 8.39 | 15 | 33 | 7 | 60 | 0 |
| 16384 | us_1D_rfft_reim-bf16 | us | 1D | rfft | reim-bf16 | — | 146.8 | 25.17 | 18 | 42 | 2 | 56 | 0 |
| 16384 | us_1D_rfft_reim | us | 1D | rfft | reim | — | 171.2 | 50.33 | 11 | 36 | 2 | 62 | 0 |
| 16384 | us_1D_rfft_fft-recur_coupled | us | 1D | rfft | fft-recur | coupled | 214.7 | 25.17 | 7 | 41 | 16 | 31 | 12 |
| 16384 | us_1D_rfft_fft-iter_coupled | us | 1D | rfft | fft-iter | coupled | 214.8 | 25.17 | 7 | 41 | 16 | 31 | 12 |
| 16384 | us_1D_rfft_conv-hlo | us | 1D | rfft | conv-hlo | — | 216.8 | 33.55 | 13 | 53 | 5 | 31 | 12 |
| 16384 | jnp_1D_rfft_jnp-fft_coupled | jnp | 1D | rfft | jnp-fft | coupled | 244.3 | 16.78 | 41 | 47 | 14 | 28 | 11 |
| 16384 | us_1D_rfft_fft-recur_reduced | us | 1D | rfft | fft-recur | reduced | 246.6 | 58.72 | 4 | 19 | 6 | 62 | 10 |
| 16384 | us_1D_rfft_fft-iter_reduced | us | 1D | rfft | fft-iter | reduced | 246.7 | 58.72 | 4 | 19 | 6 | 62 | 10 |
| 16384 | us_1D_fft_fft-iter | us | 1D | fft | fft-iter | — | 259.8 | 33.55 | 12 | 35 | 0 | 45 | 20 |
| 16384 | us_1D_fft_fft-recur | us | 1D | fft | fft-recur | — | 259.8 | 33.55 | 12 | 35 | 0 | 45 | 20 |
| 16384 | jnp_1D_rfft_jnp-fft_reduced | jnp | 1D | rfft | jnp-fft | reduced | 299.6 | 25.17 | 32 | 52 | 6 | 34 | 9 |
| 16384 | jnp_1D_rfft_jnp-fft | jnp | 1D | rfft | jnp-fft | — | 311.4 | 16.78 | 89 | 72 | 1 | 19 | 8 |
| 16384 | jnp_1D_fft_jnp-fft | jnp | 1D | fft | jnp-fft | — | 334.1 | 16.78 | 96 | 67 | 0 | 17 | 15 |
| 16384 | us_1D_rfft_direct-int8 | us | 1D | rfft | direct-int8 | — | 4210.7 | 587.20 | 106 | 10 | 90 | 0 | 0 |
| 256x256 | ★ us_2D_rfft_partial | us | 2D | rfft | partial | — | 391.9 | 100.66 | 26 | 54 | 0 | 46 | 0 |
| 256x256 | jnp_2D_rfft_jnp-fft | jnp | 2D | rfft | jnp-fft | — | 2035.3 | 741.34 | 153 | 72 | 13 | 10 | 4 |
| 256x256 | us_2D_rfft_reim | us | 2D | rfft | reim | — | 2593.7 | 1342.18 | 8 | 42 | 23 | 33 | 0 |
| 256x256 | jnp_2D_fft_jnp-fft | jnp | 2D | fft | jnp-fft | — | 3071.1 | 1207.96 | 184 | 65 | 0 | 20 | 15 |

## per (engine, N) — RAW `hlo_category` (xprof Op-Profile categories, no rollup)

**jnp_1D_fft_jnp-fft  N=256** — device 9.0 us/run, 15 runs
```
convolution fusion      65.1%  ██████████████████··········
loop fusion             21.9%  ██████······················
custom-call             12.9%  ████························
copy-start               0.2%  ····························
copy-done                0.0%  ····························
```
**jnp_1D_rfft_jnp-fft  N=256** — device 8.9 us/run, 15 runs
```
convolution fusion      66.7%  ███████████████████·········
loop fusion             23.8%  ███████·····················
custom-call              9.3%  ███·························
copy-start               0.2%  ····························
copy-done                0.0%  ····························
```
**jnp_1D_rfft_jnp-fft_coupled  N=256** — device 10.0 us/run, 15 runs
```
convolution fusion      81.7%  ███████████████████████·····
custom-call              8.1%  ██··························
reverse                  5.8%  ██··························
copy-done                2.3%  █···························
loop fusion              1.4%  ····························
slice                    0.5%  ····························
copy-start               0.1%  ····························
async-start              0.0%  ····························
async-done               0.0%  ····························
```
**jnp_1D_rfft_jnp-fft_reduced  N=256** — device 6.6 us/run, 15 runs
```
convolution fusion      39.0%  ███████████·················
slice                   15.4%  ████························
copy-done               14.0%  ████························
loop fusion             14.0%  ████························
custom-call             12.2%  ███·························
reverse                  5.0%  █···························
async-done               0.2%  ····························
copy-start               0.2%  ····························
async-start              0.1%  ····························
```
**us_1D_fft_fft-iter  N=256** — device 7.5 us/run, 15 runs
```
convolution fusion      45.9%  █████████████···············
data formatting         36.5%  ██████████··················
custom-call             15.5%  ████························
loop fusion              2.1%  █···························
```
**us_1D_fft_fft-recur  N=256** — device 7.4 us/run, 15 runs
```
convolution fusion      46.2%  █████████████···············
data formatting         36.1%  ██████████··················
custom-call             15.6%  ████························
loop fusion              2.2%  █···························
```
**us_1D_fft_pallas  N=256** — device 5.6 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_conv-hlo  N=256** — device 7.8 us/run, 15 runs
```
convolution fusion      52.7%  ███████████████·············
data formatting         15.9%  ████························
custom-call             10.6%  ███·························
copy-done                7.3%  ██··························
async-done               6.8%  ██··························
loop fusion              6.3%  ██··························
slice                    0.3%  ····························
copy-start               0.1%  ····························
async-start              0.1%  ····························
```
**us_1D_rfft_direct-int8  N=256** — device 3.4 us/run, 15 runs
```
loop fusion             68.0%  ███████████████████·········
convolution fusion      31.9%  █████████···················
copy-done                0.1%  ····························
copy-start               0.1%  ····························
```
**us_1D_rfft_fft-iter_coupled  N=256** — device 6.2 us/run, 15 runs
```
convolution fusion      47.3%  █████████████···············
loop fusion             18.0%  █████·······················
custom-call             13.0%  ████························
reverse                  9.3%  ███·························
data formatting          8.7%  ██··························
slice                    3.7%  █···························
```
**us_1D_rfft_fft-iter_reduced  N=256** — device 4.9 us/run, 15 runs
```
convolution fusion      37.7%  ███████████·················
slice                   21.8%  ██████······················
custom-call             16.6%  █████·······················
copy-done                9.0%  ███·························
loop fusion              7.8%  ██··························
reverse                  6.7%  ██··························
copy-start               0.2%  ····························
async-start              0.1%  ····························
async-done               0.1%  ····························
```
**us_1D_rfft_fft-recur_coupled  N=256** — device 6.2 us/run, 15 runs
```
convolution fusion      47.8%  █████████████···············
loop fusion             17.2%  █████·······················
custom-call             13.1%  ████························
reverse                  9.3%  ███·························
data formatting          8.8%  ██··························
slice                    3.8%  █···························
```
**us_1D_rfft_fft-recur_reduced  N=256** — device 4.9 us/run, 15 runs
```
convolution fusion      37.8%  ███████████·················
slice                   20.5%  ██████······················
custom-call             16.6%  █████·······················
copy-done               10.2%  ███·························
loop fusion              7.8%  ██··························
reverse                  6.8%  ██··························
copy-start               0.2%  ····························
async-start              0.1%  ····························
async-done               0.1%  ····························
```
**us_1D_rfft_pallas  N=256** — device 5.4 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_pallas-bf16  N=256** — device 6.0 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_pallas_reduced  N=256** — device 4.0 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_reim  N=256** — device 6.4 us/run, 15 runs
```
convolution fusion      41.4%  ████████████················
data formatting         41.2%  ████████████················
loop fusion             17.4%  █████·······················
```
**us_1D_rfft_reim-bf16  N=256** — device 6.0 us/run, 15 runs
```
convolution fusion      44.2%  ████████████················
data formatting         36.9%  ██████████··················
loop fusion             18.9%  █████·······················
```
**us_1D_rfft_reim-int8  N=256** — device 6.8 us/run, 15 runs
```
loop fusion             41.4%  ████████████················
data formatting         31.3%  █████████···················
convolution fusion      27.4%  ████████····················
```
**jnp_1D_fft_jnp-fft  N=1024** — device 24.3 us/run, 15 runs
```
convolution fusion      69.9%  ████████████████████········
data formatting         16.0%  ████························
custom-call             13.5%  ████························
loop fusion              0.6%  ····························
```
**jnp_1D_rfft_jnp-fft  N=1024** — device 23.0 us/run, 15 runs
```
convolution fusion      73.6%  █████████████████████·······
data formatting         16.8%  █████·······················
custom-call              8.1%  ██··························
loop fusion              1.5%  ····························
```
**jnp_1D_rfft_jnp-fft_coupled  N=1024** — device 17.3 us/run, 15 runs
```
convolution fusion      55.7%  ████████████████············
loop fusion             11.5%  ███·························
custom-call             10.7%  ███·························
reverse                  9.5%  ███·························
copy-done                6.7%  ██··························
data formatting          4.9%  █···························
slice                    1.1%  ····························
copy-start               0.0%  ····························
```
**jnp_1D_rfft_jnp-fft_reduced  N=1024** — device 23.4 us/run, 15 runs
```
convolution fusion      53.8%  ███████████████·············
slice                   15.3%  ████························
custom-call              7.9%  ██··························
reverse                  7.7%  ██··························
data formatting          6.5%  ██··························
loop fusion              5.1%  █···························
async-done               2.9%  █···························
non-fusion elementwise   0.8%  ····························
async-start              0.1%  ····························
```
**us_1D_fft_fft-iter  N=1024** — device 15.0 us/run, 15 runs
```
convolution fusion      47.0%  █████████████···············
data formatting         28.7%  ████████····················
custom-call             22.0%  ██████······················
async-done               1.2%  ····························
loop fusion              1.1%  ····························
async-start              0.1%  ····························
```
**us_1D_fft_fft-recur  N=1024** — device 15.0 us/run, 15 runs
```
convolution fusion      47.2%  █████████████···············
data formatting         28.7%  ████████····················
custom-call             22.0%  ██████······················
loop fusion              1.1%  ····························
async-done               0.9%  ····························
async-start              0.1%  ····························
```
**us_1D_fft_pallas  N=1024** — device 8.8 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_conv-hlo  N=1024** — device 12.4 us/run, 15 runs
```
convolution fusion      60.3%  █████████████████···········
data formatting         15.6%  ████························
custom-call             15.1%  ████························
loop fusion              5.9%  ██··························
async-done               1.5%  ····························
slice                    1.5%  ····························
async-start              0.1%  ····························
```
**us_1D_rfft_direct-int8  N=1024** — device 23.0 us/run, 15 runs
```
loop fusion             84.2%  ████████████████████████····
convolution fusion      14.8%  ████························
copy-done                0.9%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_coupled  N=1024** — device 15.1 us/run, 15 runs
```
convolution fusion      35.4%  ██████████··················
loop fusion             17.9%  █████·······················
data formatting         14.9%  ████························
custom-call             12.3%  ███·························
reverse                 10.9%  ███·························
copy-done                6.1%  ██··························
slice                    2.3%  █···························
async-start              0.1%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_reduced  N=1024** — device 16.8 us/run, 15 runs
```
convolution fusion      24.9%  ███████·····················
slice                   20.8%  ██████······················
data formatting         19.6%  ██████······················
custom-call             11.1%  ███·························
reverse                 10.8%  ███·························
loop fusion              7.2%  ██··························
async-done               4.3%  █···························
copy-done                1.1%  ····························
async-start              0.1%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_coupled  N=1024** — device 15.1 us/run, 15 runs
```
convolution fusion      35.6%  ██████████··················
loop fusion             17.6%  █████·······················
data formatting         15.0%  ████························
custom-call             12.3%  ███·························
reverse                 10.9%  ███·························
copy-done                6.2%  ██··························
slice                    2.3%  █···························
async-start              0.1%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_reduced  N=1024** — device 16.8 us/run, 15 runs
```
convolution fusion      24.9%  ███████·····················
slice                   21.0%  ██████······················
data formatting         19.7%  ██████······················
custom-call             11.1%  ███·························
reverse                 10.8%  ███·························
loop fusion              7.2%  ██··························
async-done               4.1%  █···························
copy-done                1.1%  ····························
async-start              0.1%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_pallas  N=1024** — device 7.8 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_pallas-bf16  N=1024** — device 8.1 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_pallas_reduced  N=1024** — device 35.9 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_reim  N=1024** — device 9.6 us/run, 15 runs
```
data formatting         53.3%  ███████████████·············
convolution fusion      41.5%  ████████████················
loop fusion              5.1%  █···························
copy-start               0.1%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-bf16  N=1024** — device 8.1 us/run, 15 runs
```
convolution fusion      46.9%  █████████████···············
data formatting         46.8%  █████████████···············
loop fusion              6.1%  ██··························
copy-start               0.2%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-int8  N=1024** — device 9.5 us/run, 15 runs
```
data formatting         46.0%  █████████████···············
convolution fusion      30.5%  █████████···················
loop fusion             23.5%  ███████·····················
```
**jnp_2D_fft_jnp-fft  N=64x64** — device 83.6 us/run, 15 runs
```
convolution fusion      84.6%  ████████████████████████····
custom-call             15.4%  ████························
copy-done                0.0%  ····························
copy-start               0.0%  ····························
```
**jnp_2D_rfft_jnp-fft  N=64x64** — device 51.1 us/run, 15 runs
```
convolution fusion      83.0%  ███████████████████████·····
custom-call             13.1%  ████························
loop fusion              3.9%  █···························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**us_2D_rfft_partial  N=64x64** — device 16.7 us/run, 15 runs
```
convolution fusion      52.9%  ███████████████·············
data formatting         47.1%  █████████████···············
```
**us_2D_rfft_reim  N=64x64** — device 21.0 us/run, 15 runs
```
convolution fusion      52.3%  ███████████████·············
data formatting         45.4%  █████████████···············
loop fusion              2.3%  █···························
```
**jnp_1D_fft_jnp-fft  N=4096** — device 94.4 us/run, 15 runs
```
convolution fusion      70.6%  ████████████████████········
data formatting         15.4%  ████························
custom-call             13.7%  ████························
loop fusion              0.3%  ····························
```
**jnp_1D_rfft_jnp-fft  N=4096** — device 83.7 us/run, 15 runs
```
convolution fusion      72.8%  ████████████████████········
data formatting         18.2%  █████·······················
custom-call              7.8%  ██··························
loop fusion              1.2%  ····························
```
**jnp_1D_rfft_jnp-fft_coupled  N=4096** — device 60.4 us/run, 15 runs
```
convolution fusion      57.4%  ████████████████············
loop fusion             14.0%  ████························
custom-call             10.7%  ███·························
reverse                  9.8%  ███·························
data formatting          5.4%  ██··························
non-fusion elementwise   1.5%  ····························
slice                    1.2%  ····························
```
**jnp_1D_rfft_jnp-fft_reduced  N=4096** — device 71.1 us/run, 15 runs
```
convolution fusion      48.5%  ██████████████··············
slice                   18.2%  █████·······················
custom-call              9.1%  ███·························
reverse                  8.6%  ██··························
loop fusion              5.8%  ██··························
data formatting          4.5%  █···························
copy-done                4.0%  █···························
non-fusion elementwise   1.3%  ····························
copy-start               0.0%  ····························
```
**us_1D_fft_fft-iter  N=4096** — device 57.3 us/run, 15 runs
```
convolution fusion      43.2%  ████████████················
data formatting         30.4%  █████████···················
custom-call             22.6%  ██████······················
copy-done                3.2%  █···························
loop fusion              0.5%  ····························
copy-start               0.0%  ····························
```
**us_1D_fft_fft-recur  N=4096** — device 57.2 us/run, 15 runs
```
convolution fusion      43.2%  ████████████················
data formatting         30.5%  █████████···················
custom-call             22.5%  ██████······················
copy-done                3.3%  █···························
loop fusion              0.5%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_conv-hlo  N=4096** — device 44.3 us/run, 15 runs
```
convolution fusion      59.2%  █████████████████···········
custom-call             14.6%  ████························
data formatting         13.9%  ████························
loop fusion              7.2%  ██··························
async-done               3.4%  █···························
slice                    1.6%  ····························
async-start              0.0%  ····························
```
**us_1D_rfft_direct-int8  N=4096** — device 275.5 us/run, 15 runs
```
loop fusion             91.1%  ██████████████████████████··
convolution fusion       8.9%  ██··························
```
**us_1D_rfft_fft-iter_coupled  N=4096** — device 54.0 us/run, 15 runs
```
convolution fusion      41.2%  ████████████················
loop fusion             15.7%  ████························
data formatting         15.4%  ████························
custom-call             12.1%  ███·························
reverse                 10.9%  ███·························
copy-done                2.8%  █···························
slice                    1.9%  █···························
async-start              0.0%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_reduced  N=4096** — device 60.3 us/run, 15 runs
```
convolution fusion      22.3%  ██████······················
slice                   21.3%  ██████······················
data formatting         14.3%  ████························
copy-done               12.0%  ███·························
custom-call             10.8%  ███·························
reverse                 10.1%  ███·························
loop fusion              6.8%  ██··························
async-done               2.3%  █···························
async-start              0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_coupled  N=4096** — device 53.9 us/run, 15 runs
```
convolution fusion      41.2%  ████████████················
loop fusion             15.7%  ████························
data formatting         15.4%  ████························
custom-call             12.0%  ███·························
reverse                 10.9%  ███·························
copy-done                2.8%  █···························
slice                    1.9%  █···························
async-start              0.0%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_reduced  N=4096** — device 60.3 us/run, 15 runs
```
convolution fusion      22.3%  ██████······················
slice                   21.2%  ██████······················
data formatting         14.4%  ████························
copy-done               12.1%  ███·························
custom-call             10.7%  ███·························
reverse                 10.1%  ███·························
loop fusion              6.8%  ██··························
async-done               2.3%  █···························
async-start              0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_pallas-bf16  N=4096** — device 23.7 us/run, 15 runs
```
custom-call            100.0%  ████████████████████████████
```
**us_1D_rfft_reim  N=4096** — device 31.3 us/run, 15 runs
```
data formatting         49.8%  ██████████████··············
convolution fusion      46.9%  █████████████···············
loop fusion              3.3%  █···························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-bf16  N=4096** — device 26.2 us/run, 15 runs
```
convolution fusion      55.7%  ████████████████············
data formatting         41.6%  ████████████················
loop fusion              2.6%  █···························
copy-start               0.1%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-int8  N=4096** — device 24.9 us/run, 15 runs
```
convolution fusion      44.1%  ████████████················
data formatting         41.6%  ████████████················
loop fusion             14.2%  ████························
copy-start               0.1%  ····························
copy-done                0.0%  ····························
```
**jnp_1D_fft_jnp-fft  N=8192** — device 187.4 us/run, 15 runs
```
convolution fusion      69.8%  ████████████████████········
data formatting         16.2%  █████·······················
custom-call             13.7%  ████························
loop fusion              0.3%  ····························
```
**jnp_1D_rfft_jnp-fft  N=8192** — device 176.1 us/run, 15 runs
```
convolution fusion      74.3%  █████████████████████·······
data formatting         17.3%  █████·······················
custom-call              7.3%  ██··························
loop fusion              1.1%  ····························
```
**jnp_1D_rfft_jnp-fft_coupled  N=8192** — device 123.7 us/run, 15 runs
```
convolution fusion      56.8%  ████████████████············
loop fusion             14.9%  ████························
custom-call             10.4%  ███·························
reverse                  9.4%  ███·························
data formatting          6.5%  ██··························
slice                    1.4%  ····························
non-fusion elementwise   0.7%  ····························
```
**jnp_1D_rfft_jnp-fft_reduced  N=8192** — device 140.8 us/run, 15 runs
```
convolution fusion      48.3%  ██████████████··············
slice                   19.9%  ██████······················
custom-call              9.2%  ███·························
reverse                  8.4%  ██··························
loop fusion              5.7%  ██··························
data formatting          4.7%  █···························
copy-done                3.1%  █···························
non-fusion elementwise   0.7%  ····························
copy-start               0.0%  ····························
```
**us_1D_fft_fft-iter  N=8192** — device 129.3 us/run, 15 runs
```
data formatting         38.7%  ███████████·················
convolution fusion      34.6%  ██████████··················
custom-call             20.0%  ██████······················
copy-done                6.3%  ██··························
loop fusion              0.4%  ····························
copy-start               0.0%  ····························
```
**us_1D_fft_fft-recur  N=8192** — device 129.3 us/run, 15 runs
```
data formatting         38.7%  ███████████·················
convolution fusion      34.6%  ██████████··················
custom-call             20.0%  ██████······················
copy-done                6.3%  ██··························
loop fusion              0.4%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_conv-hlo  N=8192** — device 99.5 us/run, 15 runs
```
convolution fusion      47.8%  █████████████···············
data formatting         23.2%  ███████·····················
custom-call             13.0%  ████························
copy-done                8.6%  ██··························
loop fusion              5.5%  ██··························
slice                    1.9%  █···························
copy-start               0.0%  ····························
```
**us_1D_rfft_direct-int8  N=8192** — device 1051.8 us/run, 15 runs
```
loop fusion             90.5%  █████████████████████████···
convolution fusion       8.7%  ██··························
copy-done                0.8%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_coupled  N=8192** — device 107.9 us/run, 15 runs
```
convolution fusion      39.7%  ███████████·················
loop fusion             17.1%  █████·······················
data formatting         15.4%  ████························
custom-call             11.9%  ███·························
reverse                 10.9%  ███·························
copy-done                2.6%  █···························
slice                    2.2%  █···························
async-start              0.0%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_reduced  N=8192** — device 117.2 us/run, 15 runs
```
slice                   21.6%  ██████······················
convolution fusion      21.3%  ██████······················
data formatting         14.7%  ████························
copy-done               11.9%  ███·························
custom-call             11.0%  ███·························
reverse                 10.1%  ███·························
loop fusion              6.9%  ██··························
async-done               2.4%  █···························
async-start              0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_coupled  N=8192** — device 107.8 us/run, 15 runs
```
convolution fusion      39.8%  ███████████·················
loop fusion             17.1%  █████·······················
data formatting         15.5%  ████························
custom-call             11.9%  ███·························
reverse                 10.9%  ███·························
copy-done                2.5%  █···························
slice                    2.2%  █···························
async-start              0.0%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_reduced  N=8192** — device 117.1 us/run, 15 runs
```
slice                   21.7%  ██████······················
convolution fusion      21.3%  ██████······················
data formatting         14.7%  ████························
copy-done               11.8%  ███·························
custom-call             11.0%  ███·························
reverse                 10.1%  ███·························
loop fusion              6.9%  ██··························
async-done               2.3%  █···························
async-start              0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_reim  N=8192** — device 74.6 us/run, 15 runs
```
data formatting         58.2%  ████████████████············
convolution fusion      39.2%  ███████████·················
loop fusion              2.6%  █···························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-bf16  N=8192** — device 69.0 us/run, 15 runs
```
data formatting         59.5%  █████████████████···········
convolution fusion      38.6%  ███████████·················
loop fusion              1.9%  █···························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-int8  N=8192** — device 50.4 us/run, 15 runs
```
convolution fusion      44.1%  ████████████················
data formatting         43.8%  ████████████················
loop fusion             12.1%  ███·························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**jnp_2D_fft_jnp-fft  N=128x128** — device 310.3 us/run, 15 runs
```
convolution fusion      72.3%  ████████████████████········
data formatting         26.0%  ███████·····················
custom-call              1.8%  ····························
```
**jnp_2D_rfft_jnp-fft  N=128x128** — device 228.6 us/run, 15 runs
```
convolution fusion      72.5%  ████████████████████········
data formatting         24.0%  ███████·····················
loop fusion              2.3%  █···························
custom-call              1.2%  ····························
```
**us_2D_rfft_partial  N=128x128** — device 65.0 us/run, 15 runs
```
data formatting         53.5%  ███████████████·············
convolution fusion      46.5%  █████████████···············
```
**us_2D_rfft_reim  N=128x128** — device 98.6 us/run, 15 runs
```
data formatting         81.8%  ███████████████████████·····
convolution fusion      17.3%  █████·······················
loop fusion              0.9%  ····························
```
**jnp_1D_fft_jnp-fft  N=16384** — device 334.1 us/run, 15 runs
```
convolution fusion      66.9%  ███████████████████·········
data formatting         17.5%  █████·······················
custom-call             15.4%  ████························
loop fusion              0.3%  ····························
```
**jnp_1D_rfft_jnp-fft  N=16384** — device 311.4 us/run, 15 runs
```
convolution fusion      71.7%  ████████████████████········
data formatting         18.8%  █████·······················
custom-call              8.3%  ██··························
loop fusion              1.2%  ····························
```
**jnp_1D_rfft_jnp-fft_coupled  N=16384** — device 244.3 us/run, 15 runs
```
convolution fusion      47.4%  █████████████···············
data formatting         16.8%  █████·······················
loop fusion             13.7%  ████························
custom-call             10.5%  ███·························
reverse                  9.4%  ███·························
slice                    1.5%  ····························
non-fusion elementwise   0.7%  ····························
```
**jnp_1D_rfft_jnp-fft_reduced  N=16384** — device 299.6 us/run, 15 runs
```
convolution fusion      51.6%  ██████████████··············
slice                   17.5%  █████·······················
custom-call              8.6%  ██··························
reverse                  7.8%  ██··························
data formatting          5.3%  █···························
loop fusion              5.1%  █···························
copy-done                3.4%  █···························
non-fusion elementwise   0.7%  ····························
copy-start               0.0%  ····························
```
**us_1D_fft_fft-iter  N=16384** — device 259.8 us/run, 15 runs
```
data formatting         38.2%  ███████████·················
convolution fusion      34.6%  ██████████··················
custom-call             19.7%  ██████······················
copy-done                7.1%  ██··························
loop fusion              0.4%  ····························
copy-start               0.0%  ····························
```
**us_1D_fft_fft-recur  N=16384** — device 259.8 us/run, 15 runs
```
data formatting         38.2%  ███████████·················
convolution fusion      34.6%  ██████████··················
custom-call             19.8%  ██████······················
copy-done                7.1%  ██··························
loop fusion              0.4%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_conv-hlo  N=16384** — device 216.8 us/run, 15 runs
```
convolution fusion      52.7%  ███████████████·············
data formatting         20.6%  ██████······················
custom-call             11.9%  ███·························
copy-done                8.2%  ██··························
loop fusion              4.7%  █···························
slice                    1.9%  █···························
copy-start               0.0%  ····························
```
**us_1D_rfft_direct-int8  N=16384** — device 4210.7 us/run, 15 runs
```
loop fusion             89.7%  █████████████████████████···
convolution fusion       9.8%  ███·························
copy-done                0.5%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_coupled  N=16384** — device 214.8 us/run, 15 runs
```
convolution fusion      40.9%  ███████████·················
loop fusion             16.0%  ████························
data formatting         15.5%  ████························
custom-call             12.0%  ███·························
reverse                 10.7%  ███·························
copy-done                2.5%  █···························
slice                    2.5%  █···························
async-start              0.0%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-iter_reduced  N=16384** — device 246.7 us/run, 15 runs
```
slice                   21.2%  ██████······················
data formatting         20.3%  ██████······················
convolution fusion      19.5%  █████·······················
copy-done               10.6%  ███·························
custom-call             10.4%  ███·························
reverse                  9.5%  ███·························
loop fusion              6.3%  ██··························
async-done               2.3%  █···························
async-start              0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_coupled  N=16384** — device 214.7 us/run, 15 runs
```
convolution fusion      40.9%  ███████████·················
loop fusion             16.0%  ████························
data formatting         15.5%  ████························
custom-call             12.0%  ███·························
reverse                 10.7%  ███·························
copy-done                2.6%  █···························
slice                    2.5%  █···························
async-start              0.0%  ····························
async-done               0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_fft-recur_reduced  N=16384** — device 246.6 us/run, 15 runs
```
slice                   21.2%  ██████······················
data formatting         20.3%  ██████······················
convolution fusion      19.5%  █████·······················
copy-done               10.6%  ███·························
custom-call             10.4%  ███·························
reverse                  9.5%  ███·························
loop fusion              6.3%  ██··························
async-done               2.2%  █···························
async-start              0.0%  ····························
copy-start               0.0%  ····························
```
**us_1D_rfft_reim  N=16384** — device 171.2 us/run, 15 runs
```
data formatting         61.9%  █████████████████···········
convolution fusion      35.9%  ██████████··················
loop fusion              2.2%  █···························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-bf16  N=16384** — device 146.8 us/run, 15 runs
```
data formatting         55.6%  ████████████████············
convolution fusion      42.3%  ████████████················
loop fusion              2.1%  █···························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**us_1D_rfft_reim-int8  N=16384** — device 145.4 us/run, 15 runs
```
data formatting         60.4%  █████████████████···········
convolution fusion      32.9%  █████████···················
loop fusion              6.7%  ██··························
copy-start               0.0%  ····························
copy-done                0.0%  ····························
```
**jnp_2D_fft_jnp-fft  N=256x256** — device 3071.1 us/run, 15 runs
```
convolution fusion      64.9%  ██████████████████··········
data formatting         19.8%  ██████······················
custom-call             15.2%  ████························
async-done               0.1%  ····························
async-start              0.0%  ····························
```
**jnp_2D_rfft_jnp-fft  N=256x256** — device 2035.3 us/run, 15 runs
```
convolution fusion      72.4%  ████████████████████········
loop fusion             13.4%  ████························
data formatting          9.7%  ███·························
custom-call              4.4%  █···························
async-done               0.0%  ····························
async-start              0.0%  ····························
copy-done                0.0%  ····························
copy-start               0.0%  ····························
```
**us_2D_rfft_partial  N=256x256** — device 391.9 us/run, 15 runs
```
convolution fusion      54.5%  ███████████████·············
data formatting         45.5%  █████████████···············
```
**us_2D_rfft_reim  N=256x256** — device 2593.7 us/run, 15 runs
```
convolution fusion      41.6%  ████████████················
data formatting         31.1%  █████████···················
loop fusion             22.7%  ██████······················
async-done               3.2%  █···························
copy-done                1.5%  ····························
async-start              0.0%  ····························
copy-start               0.0%  ····························
custom-call              0.0%  ····························
```

## sources
- xprof / JAX profiling: https://openxla.org/xprof/jax_profiling
- xplane proto (device_duration_ps / hlo_category / bytes_accessed / flops / memory_access_breakdown): `tensorflow.tsl.profiler.protobuf.xplane_pb2`

## fresh headline (device time, v5e, jax 0.11.1 — Pallas NOW compiles)

**Fastest USABLE engine per N** (★ in the table is device-time only; see the int8 caveat below):

| N | fastest usable | µs | why |
|---|---|---|---|
| 256 | pallas_reduced / reim-bf16 | 4.0 / 6.0 | tiny transform; VMEM kernel + fixed overhead |
| 1024 | **pallas** (VMEM-resident) | **7.8** | whole FFT in ONE fused kernel — no HBM round-trips |
| 4096 | **pallas-bf16** | **23.7** | VMEM-resident + half operand bytes |
| 8192 | **reim-bf16** | **69.0** | pallas OOMs VMEM here; bf16 halves HBM |
| 16384 | **reim-bf16** | **146.8** | same — pallas can't fit, reim-bf16 wins |
| 2D 64²/128²/256² | **partial** | 16.7 / 65 / 392 | FNO low-mode DFT; 5–11× faster than jnp/reim2 |

### Pallas (VMEM-resident) — the Phase-4 lever, now measured
With jax 0.11.1 the Pallas kernels compile on device (they failed on the old 0.6.2). Result: the
whole radix-B FFT runs as **one fused kernel** (100% `custom-call`; HBM shows `—` because the cost
model can't see inside an opaque kernel — intermediates never leave VMEM by design). This makes
**pallas the fastest engine at N=1024 (7.8 µs) and N=4096 (23.7 µs bf16)** — it removes exactly the
inter-stage corner-turn copies that cost reim ~55% of its time. It **OOMs the 16 MB scoped VMEM at
N≥8192**, so above 4096 the streaming `reim-bf16` is the usable champion. This confirms the earlier
prediction: keeping both radix stages VMEM-resident is the real HBM lever.

### ⚠ int8 accuracy caveat (why the raw ★ misleads at 256 / 8192 / 16384)
The table's ★ is **device time only**. `direct-int8` (★ at 256) and `reim-int8` (★ at 8192/16384)
are fast purely because int8 = 1 byte/component (¼ the HBM on a memory-bound transform) — but
**`reim-int8` is numerically DEAD (rel_err ≈ 1.0)**: the naive `.astype(int8)` collapses the DFT
matrix's [-1,1] entries to {-1,0,1}. `direct-int8` (scaled) is ~1e-2 but O(N²) so it only survives
at tiny N. **Discount both**; the fastest *correct* engines are pallas (≤4096) and reim-bf16 (>4096).

## FNO integration (fno_reim.py) — trains on v5e
The reim rfft is wired into a full JAX FNO (`tpu/fno_reim.py`): forward = reim `jax_rfft` (real/imag,
no complex64), keep m low modes, complex channel-mix, inverse = reim-style partial-IDFT matmul.
Spectral-conv matches a jnp.fft reference to 2e-6.  End-to-end training on the v5e (N=256, width 32,
16 modes, 4 layers, 256 samples, 200 epochs): **test rel L2 2.27 → 0.03, test_mse ~1.5e-8, 15.6 s**.
(That wall-time is host-dispatch bound for a model this small, not a paper-scale throughput number —
a 2D Darcy-scale FNO would be the apples-to-apples benchmark.)
