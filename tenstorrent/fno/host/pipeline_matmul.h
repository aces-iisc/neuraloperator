// The same operation as tile_matmul, run by the three-stage firmware instead of the one-core
// firmware.  Same inputs, same packing, same answer expected bit for bit; what differs is that
// three Tensix threads issue it and the stages overlap.
#pragma once

#include "device.h"

#include <cstdint>
#include <vector>

struct PipelineResult {
    std::vector<float> y;      // rows x cols, fp32
    uint32_t mvmuls = 0;       // issued by the math stage
    uint32_t unpacks = 0;      // issued by the unpack stage
    uint32_t packs = 0;        // issued by the pack stage
    // Simulation steps spent, for the same reason tile_matmul keeps them: deterministic, and a
    // hang shows up as a budget overrun rather than as a wedged process.  Not a time, and not
    // comparable with the one-core figure either, since the simulator retires a tile's
    // instructions eagerly and does not model overlap.
    uint64_t clocks = 0;
    bool timed_out = false;
    uint32_t done_mask = 0;    // which stages reported in; 0b111 when all three did
};

// `t` is rows x depth and `x` is depth x cols, both row-major.  The device must already be
// running the three stage images, via Device::start_pipeline.
PipelineResult pipeline_matmul(Device &device, const std::vector<float> &t,
                               const std::vector<float> &x, uint32_t rows, uint32_t depth,
                               uint32_t cols, uint32_t phases, uint32_t format);
