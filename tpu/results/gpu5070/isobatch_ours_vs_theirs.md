# Iso-batch, iso-mode inference: ours vs the reference FNO, RTX 5070

## Experiment

Forward pass of a four-layer FNO, timed at matched mode count and matched batch, so the only
thing differing between columns is the implementation.

| | |
|---|---|
| device | RTX 5070 12 GiB, driver 580.178.04, CUDA 13.0, torch 2.11.0+cu130 |
| timing | CUDA events, min of 20 reps after 8 warmups, `no_grad`, divided by batch |
| precision | float32, TF32 on (`matmul_precision="high"`, `allow_tf32=True`) |
| mode matching | by band, not `numel()`: our m=16 against their `n_modes=15`, our m=64 against their `n_modes=63` |
| points | 105 (case, batch, impl), all `ok`, no OOM and no refused shape |
| data | `isobatch_ours_vs_theirs.csv`, script `scripts/isobatch.py` |

## Columns

| column | what it is |
|---|---|
| batch | the batch both sides are timed at |
| ours | `torch_fno`, us/sample |
| reference | `neuralop` at the band-matched mode count, shipped defaults on, us/sample |
| x | reference divided by ours. Above 1.00x means ours is faster. |

## Real parameter counts

Complex parameters counted as two floats.

| case | ours | reference |
|---|---:|---:|
| Burgers / Darcy-32 / Darcy-64 | 594,841 (m=16) | 560,521 (m=15) |
| NS-128 | 67,142,721 (m=64) | 66,111,169 (m=63) |

## Burgers, grid 17x16, C=24

| batch | ours | reference | x |
|---:|---:|---:|---:|
| 16 | 40.90 | 63.56 | 1.55x |
| 64 | 10.35 | 15.74 | 1.52x |
| 128 | 7.39 | 7.99 | 1.08x |
| 256 | 6.60 | 6.35 | 0.96x |
| 396 | 6.35 | 5.88 | 0.93x |
| 512 | 6.65 | 5.87 | 0.88x |

## Darcy-32, grid 32x32, C=24

| batch | ours | reference | x |
|---:|---:|---:|---:|
| 16 | 40.77 | 62.75 | 1.54x |
| 32 | 21.23 | 31.79 | 1.50x |
| 64 | 16.18 | 19.59 | 1.21x |
| 128 | 14.36 | 17.20 | 1.20x |
| 256 | 17.79 | 22.02 | 1.24x |

## Darcy-64, grid 64x64, C=24

| batch | ours | reference | x |
|---:|---:|---:|---:|
| 8 | 84.12 | 128.05 | 1.52x |
| 16 | 53.34 | 74.12 | 1.39x |
| 32 | 47.10 | 65.55 | 1.39x |
| 64 | 56.90 | 81.04 | 1.42x |
| 128 | 77.32 | 126.45 | 1.64x |

## NS-128, grid 128x128, C=64

| batch | ours | reference | x |
|---:|---:|---:|---:|
| 1 | 1637.12 | 2956.42 | 1.81x |
| 2 | 1095.39 | 1806.82 | 1.65x |
| 4 | 831.08 | 1225.65 | 1.47x |
| 8 | 928.80 | 1288.88 | 1.39x |
| 16 | 1090.77 | 1545.34 | 1.42x |

## Best batch per case

| case | grid | batch | ours | reference | x |
|---|---|---:|---:|---:|---:|
| Burgers | 17x16 | 396 | 6.35 | 5.88 | 0.93x |
| Darcy-32 | 32x32 | 128 | 14.36 | 17.20 | 1.20x |
| Darcy-64 | 64x64 | 32 | 47.10 | 65.55 | 1.39x |
| NS-128 | 128x128 | 4 | 831.08 | 1225.65 | 1.47x |


