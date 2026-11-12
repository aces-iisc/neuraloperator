// Picking a 1-D engine by name, for the sweeps.
//
// Mirrors the dispatch half of `torch_fno/fft/transform.py`.  The three engines compute the same
// spectrum by different routes, so what separates them is the shape of the one matmul each
// issues and what the host does with its output -- never the instruction stream.  That is what
// makes them comparable on shape alone.
//
//   reim      the band's rows evaluated directly.  T is (2m x n), one pass, no recombination.
//   coupled   two real signals packed into one complex signal.  Half the matmul width, but
//             separating the two spectra needs each bin's mirror, so T is (2(2m-1) x n).
//   reduced   even and odd samples transformed at half length and recombined with a twiddle.
//             T is (2m x n/2) against an operand of twice the width.
//
// The 2-D transform is not dispatched here; it is called directly as `rfft2_low_modes`, the way
// `rfft2d.py` is in the torch port.
#pragma once

#include "core.h"

namespace fft {

enum class Engine { Reim, Coupled, Reduced };

const char *engine_name(Engine engine);

// `signal` is n x k_cols, row-major.  `phases` is 1..4; `format` is FMT_BF16 or FMT_TF32.
Spectrum transform(Device &device, Engine engine, const std::vector<float> &signal, uint32_t n,
                   uint32_t k_cols, uint32_t n_modes, uint32_t phases, uint32_t format);

// The MVMUL count the engine's shape implies, with no device involved.  This is what differs
// between the three at a given band and batch, and it is exact rather than measured.
uint64_t engine_mvmuls(Engine engine, uint32_t n, uint32_t k_cols, uint32_t n_modes,
                       uint32_t phases);

}  // namespace fft
