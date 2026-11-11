# FNO on Tenstorrent Blackhole

Porting the matmul-based real-input DFT engines (`reim`, `coupled`, `reduced`) from TPU v5e and
RTX 5070 onto Blackhole's matrix unit. Simulator now, silicon in a few weeks.

## What is here

| path | what |
| --- | --- |
| `ttsim/` | the simulator, cloned at `0afa145` (2026-11-05) |
| `ttsim/src/_out/release_bh/libttsim.so` | the Blackhole build, 273 KB |
| `harness/smoke.c` | proves the library loads, identifies itself and advances its clock |
| `reference/tt-lab/` | the one existing example of driving `libttsim.so` with no framework |
| `reference/tt-isa-doc/` | the ISA specification; `MVMUL` and the fidelity phases live here |

Build: `./make.py src/_out/release_bh/libttsim.so`. Needs g++ with C++20 and Python 3.9+,
nothing else. Under a minute.

Run the smoke test:

```
cd harness && gcc -O2 -o smoke smoke.c -ldl && ./smoke ../ttsim/src/_out/release_bh/libttsim.so
```

It should print `vendor 0x1e52  device 0xb140` -- the PCI identity of a Blackhole.

## The simulator is the chip, minus the clock

The software stack does not change between simulator and silicon. tt-metal takes
`TT_METAL_SIMULATOR=<path to libttsim.so>` and reads the architecture from the PCI device ID the
library reports, so from above there is nothing to distinguish the two. `tt-lab` carries both
backends in one file (`src/tt_backend.cpp`): `dlopen` of the library, and `/dev/tenstorrent/N`
through tt-kmd, chosen at runtime.

What does not carry over is time. From ttsim's own contract:

> `ttsim` is not cycle-accurate. Where timing is not fixed by the ISA and software
> synchronization, it may complete operations, make writes visible, and order events in any way
> they permit.

It also lists reads of cycle counters and timers as a known source of divergence from silicon,
and the Tensix model carries no instruction counter at all: counting exists only in
`src/rv64_system.cpp`, which is the RISC-V computer, not the chip. `libttsim_clock(n)` advances
the simulation by `n` steps and is deterministic, but it is not a cost measure either:
`libttsim.cpp:252` re-clocks a tile for as long as it retires at least one Tensix instruction, so
one MVMUL and sixteen MVMULs both land in the same 2000 steps.  The step count tracks scalar
RISC-V progress and collapses matrix work, so it is kept only because it makes a hang visible.
The instruction count the firmware keeps is the cost measure; see `host/roofline.h`.

So the order of work inverts relative to the TPU and GPU studies. There, time was the
measurement and correctness was the check. Here, correctness and the precision model are the
measurement, and timing waits for the board.

The `UnsupportedFunctionality` list is worth knowing but does not touch us: it is debug, scan and
BIST features, features with known hardware bugs, features without a specification, and
stochastic rounding. The matrix unit, the packers, L1 and the NoC are all modelled.

## The instruction everything rests on

```
MVMUL:  Dst[8x16] += SrcB[8x16] @ SrcA[16x16]
```

The reduction axis is 16 long, split across two lanes: lane `r0` takes `k` in 0..7 and `r1` takes
`k` in 8..15. A reduction longer than 16 is several instructions, each rounding into `Dst`
separately.

Our row transform is `Y[2m x K] = T[2m x N] @ X[N x K]`, where `K = B C N_w` is the free axis. It
lands on the instruction as:

```
        SrcA  = X, a 16 x 16 block        reduction over N, 16 at a time
        SrcB  = T, an 8 x 16 block        8 band rows at a time
        Dst   = Y, an 8 x 16 block        8 band rows x 16 signals
```

One consequence is immediate and pleasant: `RADIX` in `torch_fno/fft/core.py` is already 16, and
the matrix unit's reduction axis and `SrcA` tile are both 16. A 16-point DFT is exactly one
`SrcA` tile and one `MVMUL`. The radix-16 staged transform we already have is the right shape for
this hardware, which was not true of either earlier device.

## Fidelity phases, the new variable

Neither the TPU nor the GPU had this. The multipliers do not see a whole mantissa:

| | per phase | across all 4 phases |
| --- | --- | --- |
| `SrcA` | 5 bits | mantissa bits 9..1 -- **bit 0 is never used** |
| `SrcB` | 7 bits | mantissa bits 9..0 |

A mantissa is 10 bits plus the implicit 1. Phase bit 0 selects which half of `SrcA` is fed in,
phase bit 1 selects which half of `SrcB`, so the four phases are the four partial products:

```
   phase 0:  A_hi * B_hi        phase 2:  A_hi * B_lo
   phase 1:  A_lo * B_hi        phase 3:  A_lo * B_lo
```

Each phase is its own `MVMUL`. Full precision therefore costs 4x the multiply issues, and one
phase costs 1x at roughly 5x7 bits of input precision. That is a throughput-versus-accuracy dial
set in software, per matmul.

Two things follow for us, and both are experiments rather than guesses:

1. **Operand placement is an accuracy decision.** Whatever goes in `SrcA` permanently loses its
   mantissa LSB, even at four phases. The DFT matrix is `cos`/`sin` values in `[-1, 1]` and the
   other operand is field data, so which one can afford the lost bit is a question to measure, not
   to reason about.
2. **The widest input format is TF32 or BF16, not FP32.** `Dst` accumulates in FP32, but there is
   no FP32 multiply. Our `float64` NumPy reference and the GPU's `float32` matmuls are both
   stricter than anything this unit can do in one phase, so the band error we have been chasing on
   the GPU needs a separate error budget here.

## Where this stands

The matrix unit is running our transform. `fno/` holds the port: a tile program that does one
tiled matmul with the fidelity phase count as an input, a host driver that reaches the tile the
same way it will reach a board, and the three engines built on top. See `fno/README.md` for the
layout and the hardware details that cost the most to find.

Verified on the simulator:

- One `MVMUL` against a host reference, across operand formats, reduction depths, output row
  counts and signal widths. Every shape's instruction count matches what the shape predicts.
- All three engines against a float64 DFT over a power-of-two ladder in grid, batch and band:
  594 cells, 558 measured, and the 36 absent ones are the `reduced` engine's own limit of half
  the grid rather than anything about the hardware. `fno/results/engines_blackhole.csv`.

What it says so far, at four fidelity phases and bf16 operands:

- `coupled` costs 1.96x `reim` in multiply issues, and the per-cell ratio is 2.00x almost
  everywhere. That is the mirror rows: a band of m modes forces 2m-1 of them, and narrowing the
  band does not reduce it.
- `reduced` ties `reim` exactly in 45 of the 54 cells where both exist. It halves the reduction
  and doubles the operand width, and on a 16-wide tile those cancel. Where it loses, it is the
  small-N end, where its wider operand crosses a tile boundary `reim`'s does not.
- Accuracy is the same for `reim` and `coupled` to three digits, and `reduced` is slightly
  better, which follows from its shorter reduction.
- The fidelity ladder is clean: 2.6e-02 at one phase, 6.2e-03 at two, 4.4e-03 at four, for
  exactly 1x, 2x and 4x the multiply issues. Four phases is where bf16's own mantissa becomes
  the floor, so past that the dial buys nothing without a wider operand format.

## Next

- A 2-D cell: the column pass after the row pass, which is the same matmul against a complex
  operand and needs the three-matmul Karatsuba form rather than a new instruction.
- The staged family, which composes to `N > 16` by accumulating over the reduction axis.
- Operand placement: `SrcA` never contributes its mantissa LSB, so putting the transform matrix
  there rather than the field data is a measurable choice and has not been measured yet.
- More than one tile. Everything above is one Tensix tile of many.
- Every timing claim waits for the board.
