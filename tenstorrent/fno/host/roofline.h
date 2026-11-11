// What the measured instruction count would cost on silicon, if the matrix unit never waited.
//
// The simulator has no clock to read.  Its Tensix model is functional: `libttsim_clock` runs a
// tile for as many instructions as it can retire in one tick, so a whole matmul can land inside
// a single clock, and the tick count says nothing about how long the work takes.  Several places
// in the simulator's own source mark hardware behaviour as "not modeled (timing)".
//
// What is available instead is an exact instruction count, kept by the firmware that writes the
// instructions, and a published issue rate.  Together they give a roofline: the time the work
// would take if every MVMUL issued back to back and nothing else were on the critical path.
// That is a lower bound on the real time, not a prediction of it, and it is reported as such.
//
// The issue rate is not assumed.  Tenstorrent's documentation gives MVMUL a throughput of one
// instruction per cycle, and quotes 664 TFLOP/s for a Blackhole card; 120 Tensix x 1.35 GHz x
// 4096 FLOP per MVMUL is 663.9, so the quoted peak is exactly one MVMUL per cycle per tile and
// the two sources agree.
//
//   https://github.com/tenstorrent/tt-isa-documentation  (WormholeB0 MatrixUnit, MVMUL;
//       the Blackhole pages defer to these for the matrix unit)
//   https://docs.tenstorrent.com/aibs/blackhole/  (card comparison table)
#pragma once

#include <cstdint>

namespace roofline {

// MVMUL computes Dst[8x16] += SrcB[8x16] @ SrcA[16x16]: 2048 multiply-accumulates.
constexpr double kMvmulFlop = 2.0 * 8.0 * 16.0 * 16.0;
constexpr double kMvmulsPerCycle = 1.0;       // per Tensix, from the issue-rate table
constexpr double kAiClockHz = 1.35e9;         // "up to 1.35 GHz" on p100a / p150a / p150b
constexpr uint32_t kTensixPerCard = 120u;     // the card tables; the full ASIC carries 140
constexpr uint32_t kTensixPerAsic = 140u;

// Microseconds for `mvmuls` instructions spread over `tiles` Tensix cores.
//
// Splitting across tiles assumes the batch divides evenly and every tile holds its own operands.
// `host/multicore.cpp` does exactly that and the results come back bit-identical, so the split
// itself is measured rather than assumed; what stays assumed is that the tiles never contend for
// anything on the way in or out, which only a board can answer.
inline double micros(uint64_t mvmuls, uint32_t tiles = 1u) {
    const double cycles = double(mvmuls) / kMvmulsPerCycle;
    return cycles / (kAiClockHz * double(tiles)) * 1e6;
}

inline double tflops(uint64_t mvmuls, double micros_elapsed) {
    return double(mvmuls) * kMvmulFlop / (micros_elapsed * 1e-6) / 1e12;
}

// Peak of the whole card at a given fidelity, for putting a measured rate in context.  A phase
// is its own instruction, so four-phase bf16 runs at a quarter of the one-phase peak.
inline double card_peak_tflops(uint32_t phases, uint32_t tiles = kTensixPerCard) {
    return double(tiles) * kAiClockHz * kMvmulFlop / double(phases) / 1e12;
}

}  // namespace roofline
