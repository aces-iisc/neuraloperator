// The two packing tricks that trade matmul width for host-side recombination.
//
// Mirrors `torch_fno/fft/tricks.py`.  Both are still one matmul on the device; what they change
// is the shape of that matmul and what has to happen to its output afterwards.  The
// recombination is elementwise and stays on the host in fp32, which is bit-for-bit what the
// device's fp32 lanes would compute, so the measured error is the matrix unit's and the
// comparison between variants is about the matmul shape -- the thing that differs between them.
#pragma once

#include "core.h"

namespace fft {

// Two real signals packed into one complex signal, so the matmul is half as wide.  Separating
// the two spectra afterwards needs each bin's mirror, so the matrix carries 2(2m-1) rows rather
// than 2m: that is the scheme's real cost, and narrowing the band does not reduce it.
//
// `k_cols` must be even.  Returns re and im, each n_modes x k_cols.
Spectrum rfft_coupled_low_modes(Device &device, const std::vector<float> &signal, uint32_t n,
                                uint32_t k_cols, uint32_t n_modes, uint32_t phases,
                                uint32_t format);

// Even and odd samples transformed at half length and recombined with a twiddle, so the matrix
// is (2m x n/2) against an operand of twice the width.
//
// An (n x k) array read as (n/2 x 2k) already has [even_i | odd_i] side by side in row i, so the
// operand needs no stride-2 gather: it is the same bytes, re-shaped.  `n` must be even and the
// band cannot exceed n/2.
Spectrum rfft_reduced_low_modes(Device &device, const std::vector<float> &signal, uint32_t n,
                                uint32_t k_cols, uint32_t n_modes, uint32_t phases,
                                uint32_t format);

}  // namespace fft
