// The real 1-D transform, evaluated directly on the matrix unit.
//
// Mirrors `torch_fno/fft/rfft1d.py`.  Every function here is one call to `tile_matmul`: what
// changes between them is only which output frequencies the matrix carries.
//
// The staged radix path (`rfft_staged` in the torch port) is deliberately absent.  Its
// butterflies need a twiddle multiply between stages, which is elementwise; the matrix unit has
// no cheap elementwise lane, so on this device every stage boundary would cross back to the
// host.  Nothing here stages.
#pragma once

#include "core.h"

namespace fft {

// The DFT bins listed in `rows` of a real (n x k_cols) signal.
//
// cos and sin multiply the same operand, so they go as ONE matmul with the two stacked along the
// output axis; the matrix unit sees twice the rows per call.  Returns re and im, each
// len(rows) x k_cols.
Spectrum rfft_rows(Device &device, const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                   const std::vector<uint32_t> &rows, uint32_t phases, uint32_t format);

// The lowest `n_modes` bins only, as one direct (2*n_modes x n) matmul: one pass, no twiddle,
// no staging.  This is the `reim` engine, and the 1-D analogue of `rfft2_low_modes`.
Spectrum rfft_low_modes(Device &device, const std::vector<float> &signal, uint32_t n,
                        uint32_t k_cols, uint32_t n_modes, uint32_t phases, uint32_t format);

// One direct DFT, no staging.  `half` picks the row set rather than trimming a full result, so
// the non-redundant half costs half the matmul instead of all of it.
Spectrum rfft_direct(Device &device, const std::vector<float> &signal, uint32_t n,
                     uint32_t k_cols, bool half, uint32_t phases, uint32_t format);

}  // namespace fft
