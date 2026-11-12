# Results

One file per device.

| device | file | what it holds |
|---|---|---|
| Tenstorrent Blackhole | `tenstorrent/fno/results/RESULTS.md` | the 2-D transform kernel, and the same kernel on a CPU |
| RTX 5070 | `tpu/results/gpu5070/isobatch_ours_vs_theirs.md` | the whole FNO against the reference `neuralop`, matched band and batch |
| TPU v5e | this file | the whole FNO on four datasets |

---

## FNO inference on TPU v5e

**Experiment.** Whole four-layer FNO, trained weights, full test splits, forward only. Device
time per sample from xprof, precision `highest`, 300 epochs. `B*` is each dataset's own best
batch; the baseline is `jnp.fft` at that same batch. Source
`tpu/results/train300/docs_final.csv` part B.

**Columns.**

| column | what it is |
|---|---|
| grid | input field shape |
| num_modes | modes kept on each axis |
| C | channel width |
| objective | the loss the model was trained on |
| rel L2 | the L2 part of that objective on the test split |
| baseline | `jnp.fft` rfft2/irfft2 in complex64 |
| ours | the cos/sin partial low-mode DFT in real/imag |

### Accuracy

| dataset | grid | num_modes | C | objective | rel L2 baseline | rel L2 ours f32 | rel L2 ours bf16 |
|---|---|---|---:|:--|---:|---:|---:|
| Burgers | 17x16 | 16 | 24 | L2 | 1.036e-03 | 1.140e-03 | 2.674e-03 |
| Darcy-32 | 32x32 | 16 | 24 | H1 | 7.149e-02 | 7.149e-02 | 7.162e-02 |
| Darcy-64 | 64x64 | 16 | 24 | H1 | 4.669e-02 | 4.669e-02 | 4.657e-02 |
| Navier-Stokes | 128x128 | 64 | 64 | H1 | 1.2150e-01 | 1.2135e-01 | 1.2145e-01 |

### Device time, us/sample

| dataset | B* | baseline | ours f32 | ours bf16 | speedup f32 | speedup bf16 |
|---|---:|---:|---:|---:|---:|---:|
| Burgers | 256 | 4.43 | 3.57 | 2.85 | 1.24x | 1.56x |
| Darcy-32 | 256 | 14.88 | 7.51 | 6.25 | 1.98x | 2.38x |
| Darcy-64 | 128 | 66.51 | 22.52 | 20.65 | 2.95x | 3.22x |
| Navier-Stokes | 8 | 793.19 | 407.73 | 313.68 | 1.95x | 2.53x |

Navier-Stokes bf16 runs at B=4; every other cell holds B*.

### Against a fixed B=128 baseline

| dataset | baseline | ours bf16 | speedup |
|---|---:|---:|---:|
| Burgers | 4.78 | 2.85 | 1.68x |
| Darcy-32 | 12.33 | 6.25 | 1.97x |
| Darcy-64 | 66.46 | 20.65 | 3.22x |
| Navier-Stokes | 1327.83 | 313.68 | 4.23x |
