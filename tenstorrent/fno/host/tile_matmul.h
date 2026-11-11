// The one operation the tile performs: Y = T @ X on the matrix unit.
//
// Every transform engine in this port is one call to this, which is the point.  What separates
// reim from coupled from reduced is the shape of T and what the host does with Y afterwards,
// never the instruction stream, so the engines can be compared on the shape alone.
#pragma once

#include "device.h"

#include <cstdint>
#include <vector>

struct MatmulResult {
    std::vector<float> y;      // rows x cols, fp32, as the packer wrote it
    uint32_t mvmuls = 0;       // MVMUL instructions the tile issued, fidelity phases included
    uint32_t unpacks = 0;      // UNPACR: operand tiles moved into SrcA and SrcB
    uint32_t packs = 0;        // PACR: result rows moved out of Dst
    // Simulation steps spent.  Not a time and not a work measure either: one step runs a tile
    // for as many instructions as it can retire, so a whole matmul can land in a single step.
    // Kept only because it is deterministic and makes a hang visible.
    uint64_t clocks = 0;
    bool timed_out = false;
};

// `t` is rows x depth and `x` is depth x cols, both row-major.  `format` is FMT_BF16 or
// FMT_TF32 and applies to both operands; `phases` is 1..4 and sets how much of each mantissa
// the multipliers see.
MatmulResult tile_matmul(Device &device, const std::vector<float> &t, const std::vector<float> &x,
                         uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases,
                         uint32_t format);

// The instruction count the shape implies, independent of running anything.  The tile pads to
// whole tiles, so this is a step function of the band and the batch, not a smooth one.
inline uint64_t expected_mvmuls(uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases) {
    return uint64_t((rows + 7) / 8) * ((depth + 15) / 16) * ((cols + 15) / 16) * phases;
}
