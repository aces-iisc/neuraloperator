// The three real-input transform engines, on Blackhole's matrix unit.
//
// These are the same three the TPU and GPU studies compare, ported without changing what they
// compute.  Each takes a real (n x k) block of signals and returns the lowest `n_modes` bins as
// a real and an imaginary half, and each is one matmul of a stacked cos/sin matrix followed by
// an elementwise fix-up:
//
//   reim      the band's rows evaluated directly.  T is (2m x n), one pass, no recombination.
//   coupled   two real signals packed into one complex signal.  Half the matmul width, but
//             separating the two spectra needs each bin's mirror, so T is (2(2m-1) x n).
//   reduced   even and odd samples transformed at half length and recombined with a twiddle.
//             T is (2m x n/2) against an operand of twice the width.
//
// The matmul runs on the matrix unit.  The recombination is elementwise and stays on the host in
// fp32, which is bit-for-bit what the device's fp32 lanes would compute, so the measured error
// is the matrix unit's and the comparison between engines is about the matmul shape -- which is
// the thing that differs between them.
#pragma once

#include "device.h"
#include "tile_matmul.h"

#include <cstdint>
#include <string>
#include <vector>

enum class Engine { Reim, Coupled, Reduced };

const char *engine_name(Engine engine);

struct Spectrum {
    std::vector<float> re, im;     // each n_modes x k_cols
    uint32_t mvmuls = 0;
    uint32_t unpacks = 0;
    uint32_t packs = 0;
    uint64_t clocks = 0;
    bool timed_out = false;
    std::string error;             // non-empty if the shape is one this engine cannot take
};

// `signal` is n x k_cols, row-major.  `phases` is 1..4; `format` is FMT_BF16 or FMT_TF32.
Spectrum transform(Device &device, Engine engine, const std::vector<float> &signal, uint32_t n,
                   uint32_t k_cols, uint32_t n_modes, uint32_t phases, uint32_t format);

// The MVMUL count the engine's shape implies, with no device involved.  This is what differs
// between the three at a given band and batch, and it is exact rather than measured.
uint64_t engine_mvmuls(Engine engine, uint32_t n, uint32_t k_cols, uint32_t n_modes,
                       uint32_t phases);

// The exact transform in float64, for the first `n_modes` bins: what every engine is judged
// against, as in the TPU and GPU sweeps.
void exact_spectrum(const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                    uint32_t n_modes, std::vector<double> &re, std::vector<double> &im);

// Largest absolute error over the largest exact magnitude, across both halves together.
double spectrum_error(const Spectrum &got, const std::vector<double> &re,
                      const std::vector<double> &im);
