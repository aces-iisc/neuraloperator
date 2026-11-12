# The transform on Tenstorrent Blackhole

No trained weights. The transform kernel runs on random fields against a float64 reference,
through the ttsim simulator. Operands are bf16; L1 holds fp32, the unpacker narrows into
SrcA/SrcB, and Dst accumulates fp32.

## Columns

| column | what it is |
|---|---|
| field / grid | input shape |
| batch | how many fields or signals are transformed in one call |
| num_modes | modes kept per axis. One number means the same on both; `32/8` means 32 on rows and 8 on columns. |
| matmul shape | `M x K x N` of each matmul the pass issues |
| rel err | relative error against the exact partial DFT in float64 |
| sim wallclock | host time for the simulated run: x86 interpreting RISC-V and Tensix. Not device time. Three-run average. |

## 2-D transform

**Experiment.** `rfft2_low_modes` on 10 shapes, `host/check_rfft2.cpp`. One stacked real row
matmul, then three Karatsuba complex column matmuls.

```sh
cd tenstorrent/fno && ./build.sh
./_out/check_rfft2 ../ttsim/src/_out/release_bh/libttsim.so
```

| field | batch | num_modes | matmul shape | rel err | sim wallclock |
|---|---:|---|---|---:|---:|
| 16x16 | 1 | 8 | 16x16x16 + 3x(8x16x8) | 4.396e-02 | 122 us |
| 32x32 | 1 | 16 | 32x32x32 + 3x(16x32x16) | 5.298e-02 | 259 us |
| 64x64 | 1 | 32 | 64x64x64 + 3x(32x64x32) | 6.143e-02 | 1966 us |
| 64x64 | 1 | 8 | 16x64x64 + 3x(8x64x8) | 5.446e-02 | 314 us |
| 64x64 | 1 | 32/8 | 64x64x64 + 3x(8x64x32) | 7.209e-02 | 1003 us |
| 64x64 | 1 | 8/32 | 16x64x64 + 3x(32x64x8) | 5.527e-02 | 495 us |
| 32x32 | 1 | 32/16 | 64x32x32 + 3x(16x32x32) | 5.934e-02 | 429 us |
| 32x64 | 1 | 16/32 | 32x32x64 + 3x(32x64x16) | 4.736e-02 | 581 us |
| 32x32 | 8 | 16 | 32x32x256 + 3x(16x32x128) | 5.333e-02 | 1531 us |
| 64x64 | 8 | 16 | 32x64x512 + 3x(16x64x128) | 5.672e-02 | 4343 us |

## 1-D transform

**Experiment.** `host/sweep.cpp` at half band on each grid, batch 64, one matmul per call. Three
ways to get a real signal's spectrum out of one matmul: `reim` stacks cos and sin; `coupled`
packs two signals into one complex transform; `reduced` splits the signal into even and odd
halves and recombines on the host.

```sh
./_out/sweep ../ttsim/src/_out/release_bh/libttsim.so results/engines_blackhole.csv
```

| grid | batch | num_modes | engine | matmul shape | matmuls | rel err |
|---:|---:|---:|---|---|---:|---:|
| 16 | 64 | 8 | reim | 16x16x64 | 8 | 3.074e-02 |
| 16 | 64 | 8 | coupled | 30x16x64 | 16 | 3.074e-02 |
| 16 | 64 | 8 | reduced | 16x8x128 | 16 | 3.204e-02 |
| 32 | 64 | 16 | reim | 32x32x64 | 32 | 3.061e-02 |
| 32 | 64 | 16 | coupled | 62x32x64 | 64 | 3.061e-02 |
| 32 | 64 | 16 | reduced | 32x16x128 | 32 | 2.953e-02 |
| 64 | 64 | 32 | reim | 64x64x64 | 128 | 2.922e-02 |
| 64 | 64 | 32 | coupled | 126x64x64 | 256 | 2.922e-02 |
| 64 | 64 | 32 | reduced | 64x32x128 | 128 | 2.782e-02 |
| 128 | 64 | 64 | reim | 128x128x64 | 512 | 2.318e-02 |
| 128 | 64 | 64 | coupled | 254x128x64 | 1024 | 2.318e-02 |
| 128 | 64 | 64 | reduced | 128x64x128 | 512 | 2.318e-02 |
| 256 | 64 | 128 | reim | 256x256x64 | 2048 | 2.962e-02 |
| 256 | 64 | 128 | coupled | 510x256x64 | 4096 | 2.962e-02 |
| 256 | 64 | 128 | reduced | 256x128x128 | 2048 | 2.900e-02 |

`matmuls` is the instruction count the firmware reports. The sweep run did not record sim
wallclock, so that column is absent here; the full sweep is 783 rows over five grids, three
batches and every band, in `engines_blackhole.csv`.

## The same transform on a CPU

**Experiment.** Same algorithm and same FLOP count as the 2-D table, reimplemented in numpy
fp32. AMD Ryzen 7 9700X, 8 cores, Zen 5, AVX-512, numpy 2.4.6 on OpenBLAS 0.3.31. Fastest of
50 runs after 10 warmups, all 8 cores. `rfft2 + slice` is `numpy.fft.rfft2` on the full field
followed by a band slice.

```sh
.venv/bin/python tools/cpu_reference.py results/cpu_rfft2_allcores.csv
CPU_REF_THREADS=1 .venv/bin/python tools/cpu_reference.py results/cpu_rfft2_1core.csv
```

| field | batch | num_modes | ours us | rfft2 + slice us | rel err |
|---|---:|---|---:|---:|---:|
| 16x16 | 1 | 8 | 21.63 | 8.27 | 1.500e-07 |
| 32x32 | 1 | 16 | 27.11 | 10.15 | 1.402e-07 |
| 64x64 | 1 | 32 | 71.11 | 15.90 | 1.808e-07 |
| 64x64 | 1 | 8 | 26.95 | 15.88 | 1.547e-07 |
| 64x64 | 1 | 32/8 | 49.03 | 15.81 | 1.861e-07 |
| 64x64 | 1 | 8/32 | 47.68 | 15.83 | 1.795e-07 |
| 32x32 | 1 | 32/16 | 34.98 | 10.07 | 1.418e-07 |
| 32x64 | 1 | 16/32 | 48.84 | 12.15 | 1.443e-07 |
| 32x32 | 8 | 16 | 34.02 | 33.43 | 1.853e-07 |
| 64x64 | 8 | 16 | 57.65 | 98.99 | 2.467e-07 |
| 64x64 | 64 | 32 | 254.88 | 1,030.08 | 2.497e-07 |
| 128x128 | 64 | 64 | 999.03 | 3,569.16 | 3.491e-07 |
| 128x128 | 256 | 64 | 10,301.80 | 32,380.74 | 3.506e-07 |
| 256x256 | 64 | 128 | 8,526.92 | 24,072.73 | 4.956e-07 |

Single-core times are in `cpu_rfft2_1core.csv`.

## Files

| file | what it holds |
|---|---|
| `engines_blackhole.csv` | the full 1-D sweep, 783 rows |
| `scaling_blackhole.csv` | multi-core scaling |
| `cpu_rfft2_allcores.csv` | the CPU table, 8 cores |
| `cpu_rfft2_1core.csv` | the same, 1 core |
