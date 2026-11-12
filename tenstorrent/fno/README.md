# The transform engines on Blackhole's matrix unit

The three real-input transform engines from `torch_fno/fft`, running on a Blackhole Tensix tile
through the `ttsim` simulator. Same algorithm, same packing schemes, same stacked cos/sin matmul.

## What this measures, and what it cannot

The TPU and GPU studies measured time and checked correctness. This one is the other way round,
because of what the simulator is. From `ttsim`'s own contract:

> `ttsim` is not cycle-accurate. Where timing is not fixed by the ISA and software
> synchronization, it may complete operations, make writes visible, and order events in any way
> they permit.

The Tensix model carries no instruction or cycle counter either, and the simulator's clock is
not a work measure in disguise: `libttsim_clock` runs a tile for as many instructions as it can
retire in one tick, so one MVMUL and sixteen MVMULs both complete in the same 2000 clocks. The
`clocks` column makes a hang visible and nothing more.

What the simulator answers exactly, and better than silicon can, is the arithmetic: every
product, every rounding, every fidelity phase, bit for bit. It also gives an exact instruction
count, because the firmware counts what it issues. So the sweep reports

| column | meaning |
| --- | --- |
| `rel_err` | largest error over the band, against a float64 DFT |
| `mvmuls` | MVMUL instructions issued, fidelity phases included |
| `unpacks`, `packs` | UNPACR and PACR issued, so the non-FPU work is visible |
| `predicted` | what the engine's shape implies, computed without the device |
| `us_1tile`, `us_card` | projected matrix-unit issue time; see below |

## Time, and what the projection is worth

There is no measured time here, but there is a defensible floor on it. MVMUL has a published
throughput of one instruction per cycle per Tensix, and Blackhole runs at up to 1.35 GHz. The
two sources agree with each other: 120 Tensix x 1.35 GHz x 4096 FLOP per MVMUL is 663.9
TFLOP/s, which is exactly the 664 TFLOPS the card table quotes. So

```
us_1tile = mvmuls / 1.35 GHz            one Tensix, issuing back to back
us_card  = us_1tile / 120               the same work split over a card
```

This is a roofline, not a prediction. It counts multiply issue alone: no kernel launch, no
operand movement, no stall, and `us_card` further assumes the batch splits evenly over tiles,
which our single-tile firmware does not do. Real time is longer, and how much longer is the
question the board answers.

**Do not divide it against the TPU and GPU sweeps.** Those are measured wall times at shapes
where a 5070 runs at under 5% of its bf16 peak, so they are launch-bound and the ratio would
measure launch overhead rather than either machine. One Tensix tile holds 1.5 MB of L1, which
caps the transform at N = 256 on a single tile, and that ceiling sits well below the size at
which a GPU becomes compute-bound. A fair comparison needs the multi-tile program, or a board.

What the projection is good for is absolute statements about the hardware. The fidelity dial,
for one, costs exactly what the instruction count says:

| phases | card peak | our measured `rel_err` |
| --- | --- | --- |
| 1 | 664 TFLOP/s | 2.6e-02 |
| 2 | 332 TFLOP/s | 6.2e-03 |
| 4 | 166 TFLOP/s | 4.4e-03 |

## Across the card

`scale` runs the same transform on 1, 2, 4 ... 120 Tensix and checks every split against the
one-tile answer. A card is 120 usable Tensix, not 140: Blackhole harvests two of its fourteen
columns, and the ARC turns those tiles' clocks off at boot. Nothing here hardcodes that. A TLB
window names its tile by a *virtual* coordinate and the NOC translation maps the virtual Tensix
columns onto the surviving physical ones in order, so the driver only needs the count of live
columns, which it reads from the router's own mask.

The split is over the batch, and nothing crosses between tiles: every tile holds the whole
transform matrix and its own slice of the signals. Results:

- Scaling is **exactly linear** to the ceiling. Speedup is `min(tiles, column tiles)` with no
  loss anywhere below it, and every split is bit-for-bit identical to the one-tile answer.
- The ceiling is the batch. The matrix unit cannot take less than a 16-wide column tile, so a
  shape offers `ceil(B / 16)` pieces. **Filling 120 Tensix needs a batch of at least 1920.**
  At B = 256 the card stops at 16 tiles however many are offered.
- The only sublinear point is integer leftovers: 256 pieces over 120 tiles leaves some tiles
  holding three and some two, so the speedup is 85x rather than 120x.
- Splitting also lifts the single-tile L1 ceiling. Shapes too big for one tile's 1.5 MB run
  once divided, which is why some rows have no single-tile column to check against.

| column | meaning |
| --- | --- |
| `tiles_offered`, `tiles_used` | how many were allowed, and how many got work |
| `mvmuls_critical` | the busiest tile's count: what the card waits for |
| `speedup`, `efficiency` | against one tile, and per tile actually used |
| `padding_overhead` | total work split over total work on one tile |
| `max_abs_diff` | against the one-tile result; 0 everywhere it could be checked |

## Three kernels, and what the split actually cost

The one-core firmware runs on RISCV B and issues unpack, multiply and pack in sequence. The
three-stage firmware in `fw/pipeline.cpp` puts each of those on its own Tensix thread, which is
how tt-metal kernels are organised and the only arrangement in which the stages can overlap:

| thread | stage | issues |
| --- | --- | --- |
| T0 | unpack | `UNPACR` into SrcA and SrcB, one pair per reduction step |
| T1 | math | `MVMUL`, once per fidelity phase, accumulating in Dst |
| T2 | pack | two `PACR` per 8x16 output block |

The overlap comes from hardware. SrcA and SrcB are double banked: the unpacker fills one bank
while the matrix unit reads the other. So T0 can be a reduction step ahead of T1, and all
software has to supply is a credit that stops it getting further than the banks allow. That is
what the Sync Unit's semaphores do here, one pair per handoff. Dst is left single buffered, so
pack does not yet overlap math; Dst has 512 fp32 rows and a block uses eight, so the room is
there, and keeping Dst addressing identical to the one-core version is what makes the comparison
below mean something.

`check_pipeline` runs both firmwares on the same inputs and compares. Twelve shapes, chosen so
each loop bound fails on its own if it is wrong, and every one comes back **bit-identical with
matching MVMUL, UNPACR and PACR counts**. That is the strongest claim available: the split
changed who issues each instruction, and changed nothing about the arithmetic.

Two things the split needed that the one-core version did not, both found by running it:

- **Per-thread state is per thread.** The AddrMod table that `SETC16` writes and the address
  counters that `SETADCXX` bounds belong to the issuing thread, not the tile. Only the `cfg()`
  registers are shared. A pack stage inheriting another thread's setup finds the packer's datum
  count at its reset value of one instead of sixteen.
- **A base address must be written by an instruction, not by a store.** With one core, a store
  to a configuration register followed by the instruction that reads it behaves itself. With
  three, each core runs its own loop and gets ahead of its own instruction stream. With plain
  stores the pack stage wrote four blocks' results to the addresses of four *later* blocks and
  left the first four never written. `RMWCIB` carries the value as an immediate from inside the
  instruction stream and cannot be overtaken.

That second one is a bug the one-core firmware has too, latent. Auto TTSync is the hardware
interlock that makes the plain store safe, it covers T0/T1/T2 and never RISCV B, and it cannot
be switched on here because the simulator has no such register. So the one-core path depends on
the core and the coprocessor staying in step, which is true in the simulator and not promised on
a board.

**What this cannot show is a speedup.** There is no time to measure, and the simulator does not
model the stalls either: it ignores `STALLWAIT`'s block mask outright and treats most of its
conditions as already met. What it does model, and did enforce, is the semaphore protocol -- it
rejected the first version of this firmware for a `SEMWAIT` whose block mask left out the Sync
Unit, which would have let the following `SEMGET` take a credit that was not there. So the
structure is verified and the overlap is argued from the ISA, not measured.

## The instruction everything rests on

```
MVMUL:  Dst[8x16] += SrcB[8x16] @ SrcA[16x16]
```

The reduction axis is 16 long. Our row transform is `Y[2m x K] = T[2m x N] @ X[N x K]`, so it
lands on the instruction as `SrcB` taking eight rows of `T`, `SrcA` taking a 16x16 block of the
signals, and `Dst` taking eight band rows by sixteen signals. `RADIX` in `torch_fno/fft/core.py`
is already 16, and so are the matrix unit's reduction axis and `SrcA` tile: a 16-point DFT is
exactly one `SrcA` tile and one `MVMUL`.

## Fidelity phases: the axis the earlier devices did not have

The multipliers do not see a whole mantissa. Each pass feeds five bits of `SrcA` against seven
bits of `SrcB`, and four passes over the same operands are needed for the full product. Each
pass is its own `MVMUL`, so precision is a throughput dial set per matmul. Measured here:

| phases | `rel_err` | MVMULs |
| --- | --- | --- |
| 1 | 2.2e-02 | 1x |
| 2 | 3.6e-03 | 2x |
| 4 | 1.8e-04 | 4x |

`SrcA` never contributes its mantissa LSB even at four phases, so which operand goes where is an
accuracy decision. The transform matrix and the field data are not interchangeable in that slot.

## Two axes

`host/fft/` mirrors `torch_fno/fft/` file for file, so the two ports read side by side. The 1-D
engines moved there unchanged -- the sweep's 594 cells come back byte-identical -- and
`rfft2d.cpp` is the new half.

`rfft2_low_modes` keeps the lowest `row_modes x col_modes` bins of a real
`(n_rows x n_cols x batch)` field, and nothing outside that band is ever computed. The rows are
real, so cos and sin stack into **one** matmul. The columns are complex, so they take **three**
(Gauss/Karatsuba, `p1 = ac`, `p2 = bd`, `p3 = (a+b)(c+d)`), which is what the torch port does and
why `complex_matmul` exists at all: no 1-D variant needs it.

Four matmuls in total. For a 64x64 field with half the modes kept, at one phase, that is 128
MVMUL for the rows and 96 for the columns.

The one place this port diverges from the GPU is how the column pass is **grouped**. The torch
version contracts the middle axis with an einsum, `dw,awt->adt`, so it never transposes -- on a
GPU a transpose is real data movement. There is no einsum here: the device performs `Y = T @ X`
and nothing else, so the contracted axis has to lead. Folding that axis swap into `pack_tiles`,
which reorders every operand into block-major on the way to L1 anyway, costs nothing and makes
the batch `row_modes` times longer, so the padding is paid once instead of per row mode. The
alternative -- one matmul per row mode, keeping the torch axis order -- issues **1536** MVMUL
where this issues **96** for the same arithmetic.

What is *not* ported is `rfft2_full`, because its row pass is the staged radix rfft. Staging needs
a twiddle multiply between butterfly stages; that is elementwise, the matrix unit has no cheap
elementwise lane, and every stage boundary would cross back to the host. Nothing here stages.

`check_rfft2` runs twelve shapes against the exact partial DFT in float64. All twelve match
`rfft2_mvmuls` exactly, which is the structural check. The accuracy is reported rather than
asserted: with `FNO_STAGES=1` it also runs the row pass alone, and the 2-D error comes out at
**2.0-2.6x** the row-pass error in every case -- two sequential roundings, the second amplified
a little by Karatsuba's `c - a - b` cancellation. The ratio holds across band width, batch and
aspect, which is what distinguishes it from a loop-bound mistake.

## Layout

| path | what |
| --- | --- |
| `include/tensix.h` | instruction encodings, config registers, the L1 map the host and tile agree on |
| `fw/kernel.cpp` | the tile program: one tiled matmul, with the fidelity phase count as an input |
| `fw/entry.S`, `fw/kernel.ld` | BRISC starts at L1 address zero, so the image is linked flat |
| `host/device.{h,cpp}` | the simulator as a PCIe device: BAR windows, L1, tile boot. Carries over to silicon unchanged |
| `host/numerics.{h,cpp}` | operand formats, tile layouts, the float64 reference |
| `host/tile_matmul.{h,cpp}` | the one device operation every transform is built from |
| `host/fft/core.{h,cpp}` | row sets, restricted DFT matrices, the Karatsuba product, the float64 references |
| `host/fft/rfft1d.{h,cpp}` | the real 1-D transform: `rfft_rows`, `rfft_low_modes`, `rfft_direct` |
| `host/fft/tricks.{h,cpp}` | the coupled and reduced packings |
| `host/fft/rfft2d.{h,cpp}` | the 2-D transform, band-limited on both axes |
| `host/fft/transform.{h,cpp}` | picking a 1-D engine by name, for the sweeps |
| `host/check_mvmul.cpp` | the matrix unit against a host reference, shape by shape |
| `host/probe_dst.cpp` | Dst through the debug array port, for when the packer is the suspect |
| `host/sweep.cpp` | the ladder, to CSV |
| `fw/pipeline.cpp` | the same matmul split across three Tensix threads, one source per stage |
| `host/pipeline_matmul.{h,cpp}` | the three-stage form of one matmul, driven by an epoch counter |
| `host/check_pipeline.cpp` | three stages against one core, bit for bit |
| `host/check_rfft2.cpp` | the 2-D transform against the exact partial DFT in float64 |
| `tools/cfgreg.py` | which config register a named field lives in, read from the simulator's own description |

## Building and running

The firmware needs Tenstorrent's RISC-V toolchain; set `SFPI` if it is not in `../toolchain/sfpi`.

```sh
./build.sh
_out/check_mvmul     ../ttsim/src/_out/release_bh/libttsim.so  # the matrix unit, verified
_out/check_pipeline  ../ttsim/src/_out/release_bh/libttsim.so  # three stages vs one core
_out/check_rfft2     ../ttsim/src/_out/release_bh/libttsim.so  # the 2-D transform, verified
_out/sweep           ../ttsim/src/_out/release_bh/libttsim.so  # the engines, to results/
_out/scale           ../ttsim/src/_out/release_bh/libttsim.so  # across the Tensix, to results/
```

Every axis is a power of two, so columns line up across grids and a missing cell is a shape that
does not exist or does not fit rather than a gap in the sampling. The ladders are overridable:
`FNO_GRIDS`, `FNO_BATCHES`, `FNO_BANDS`, `FNO_PHASES`, and `FNO_TF32=1` for tf32 operands.

## Notes for the silicon bring-up

- The software stack does not change. `host/device.cpp` reaches the tile through BAR windows and
  a soft-reset register; on a board the same calls go to a real mapping instead of to the library.
- `libttsim_tile_rd_bytes`/`wr_bytes` are illegal on Blackhole. The BAR path is the only one.
- An unpack into `SrcA` carries a fixed four-row destination offset that the unpacker subtracts
  back off, so `THCON_SEC0_REG5_Dest_cntx0_address` must be exactly 64 for a tile to land at row 0.
- An unpack moves as many datums as the X counter bounds allow, not one row.
- The packer's datum count is per read interface and must be sixteen; four interfaces then cover
  four rows, so an 8x16 block is two packs sharing one output tile.
- `PCK_EDGE_OFFSET_SEC0_mask` gates every column the packer writes. Left at zero it writes zeros
  without reading Dst, which looks exactly like an arithmetic failure and is not one.
- `MVMUL`'s `instr_mod19` selects broadcast: 1 is a matrix times a vector, which is what a matvec
  kernel wants and not what a matmul wants. The full eight-row form is 0.
