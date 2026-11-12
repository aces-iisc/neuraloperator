// The real 2-D transform, band-limited on both axes.
//
// Mirrors `torch_fno/fft/rfft2d.py`.  Only `rfft2_low_modes` is ported: it is the variant the
// FNO layer actually calls and the one the GPU and TPU studies measured as the champion.
// `rfft2_full` is not here, because its row pass is the staged radix rfft and nothing in this
// port stages -- see the note at the top of `rfft1d.h`.
#pragma once

#include "core.h"

namespace fft {

// The lowest `row_modes x col_modes` bins of a real (n_rows x n_cols x batch) field, as a direct
// partial DFT on each axis.  No full transform, no twiddle, no complex operands: the rows are
// real so they cost two stacked matmuls in one pass, and the columns are complex so they cost
// three (Karatsuba).  Four matmuls in total, and nothing outside the band is ever computed.
//
// `torch_fno` takes a `ratio`; the modes are given explicitly here because the sweeps drive the
// band directly.  `ratio` maps to `max(1, int(n * ratio))` on each axis.
//
// Returns re and im, each row_modes x col_modes x batch.
//
// -- how the column pass is grouped ----------------------------------------------------------
// The torch version contracts the *middle* axis with an einsum, `dw,awt->adt`, so it needs no
// transpose: on a GPU or TPU a transpose is real data movement, so avoiding it is the point.
// There is no einsum here.  The device performs one operation, `Y = T @ X`, so the contracted
// axis has to come first, and there are two ways to arrange that:
//
//   (a) one matmul per retained row mode, keeping the torch axis order exactly.  Every call then
//       pads the band up to TILE_M and the batch up to TILE_N again.
//   (b) fold the axis swap into the host-side repack and issue ONE wide matmul whose batch is
//       row_modes times longer.
//
// (b) is what runs here.  `pack_tiles` already reorders every operand into block-major on the
// way to L1, so the swap rides along inside a copy that happens regardless -- the cost that
// makes a transpose worth avoiding on the other two devices does not exist on this one.  The
// arithmetic is identical; the instruction count is not.  At 64x64 with half the modes kept and
// one phase, (a) issues 1536 MVMUL and (b) issues 96.
Spectrum rfft2_low_modes(Device &device, const std::vector<float> &field, uint32_t n_rows,
                         uint32_t n_cols, uint32_t batch, uint32_t row_modes, uint32_t col_modes,
                         uint32_t phases, uint32_t format);

// The MVMUL count the shape implies, with no device involved: one row pass plus three column
// passes, each padded to whole tiles.
uint64_t rfft2_mvmuls(uint32_t n_rows, uint32_t n_cols, uint32_t batch, uint32_t row_modes,
                      uint32_t col_modes, uint32_t phases);

}  // namespace fft
