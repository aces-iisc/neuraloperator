#include "pipeline_matmul.h"
#include "numerics.h"
#include "../include/tensix.h"

#include <stdexcept>

namespace {

constexpr uint64_t kClockBudget = 200ull * 1000 * 1000;
constexpr uint32_t kAllStages = 0x7u;

uint32_t command_field(uint32_t index) {
    return L1_COMMAND_ADDR + index * 4u;
}

}  // namespace

PipelineResult pipeline_matmul(Device &device, const std::vector<float> &t,
                               const std::vector<float> &x, uint32_t rows, uint32_t depth,
                               uint32_t cols, uint32_t phases, uint32_t format) {
    if (phases < 1 || phases > 4) {
        throw std::runtime_error("fidelity phases must be 1..4");
    }
    const uint32_t row_tiles = tiles_for(rows, TILE_M);
    const uint32_t depth_tiles = tiles_for(depth, TILE_K);
    const uint32_t col_tiles = tiles_for(cols, TILE_N);

    const std::vector<uint8_t> t_bytes = pack_tiles(t, rows, depth, TILE_M, TILE_K, FMT_FP32);
    const std::vector<uint8_t> x_bytes = pack_tiles(x, depth, cols, TILE_K, TILE_N, FMT_FP32);
    const size_t y_bytes = size_t(row_tiles) * col_tiles * TILE_M * TILE_N * 4;

    if (L1_MATRIX_ADDR + t_bytes.size() > L1_OPERAND_ADDR ||
        L1_OPERAND_ADDR + x_bytes.size() > L1_OUTPUT_ADDR ||
        L1_OUTPUT_ADDR + y_bytes > L1_SIZE) {
        throw std::runtime_error("shape does not fit in L1");
    }

    device.write_l1(L1_MATRIX_ADDR, t_bytes.data(), uint32_t(t_bytes.size()));
    device.write_l1(L1_OPERAND_ADDR, x_bytes.data(), uint32_t(x_bytes.size()));

    // Poison the output, for the same reason the one-core path does: a packer that never ran
    // reads back as the pattern rather than as zeros, which distinguishes a stage that stalled
    // from arithmetic that went wrong.
    const std::vector<uint8_t> poison(y_bytes, 0xCD);
    device.write_l1(L1_OUTPUT_ADDR, poison.data(), uint32_t(poison.size()));

    const uint32_t epoch = device.read_l1_u32(command_field(CMDF_EPOCH)) + 1u;

    device.write_l1_u32(command_field(CMDF_ROW_TILES), row_tiles);
    device.write_l1_u32(command_field(CMDF_DEPTH_TILES), depth_tiles);
    device.write_l1_u32(command_field(CMDF_COL_TILES), col_tiles);
    device.write_l1_u32(command_field(CMDF_PHASES), phases);
    device.write_l1_u32(command_field(CMDF_FORMAT), format);
    device.write_l1_u32(command_field(CMDF_DOORBELL), CMD_MATMUL);
    device.write_l1_u32(command_field(CMDF_DONE_MASK), 0u);
    // The epoch goes last.  Each stage is spinning on it, so until it changes none of them can
    // see a half-written command.
    device.write_l1_u32(command_field(CMDF_EPOCH), epoch);

    PipelineResult result;
    result.clocks = device.clock_until(
        [&] {
            return device.read_l1_u32(command_field(CMDF_DONE_MASK)) == kAllStages;
        },
        kClockBudget);
    result.done_mask = device.read_l1_u32(command_field(CMDF_DONE_MASK));
    if (result.clocks == 0) {
        result.timed_out = true;
        return result;
    }

    result.mvmuls = device.read_l1_u32(command_field(CMDF_MVMULS));
    result.unpacks = device.read_l1_u32(command_field(CMDF_UNPACKS));
    result.packs = device.read_l1_u32(command_field(CMDF_PACKS));
    std::vector<uint8_t> out(y_bytes);
    device.read_l1(L1_OUTPUT_ADDR, out.data(), uint32_t(out.size()));
    result.y = unpack_tiles_f32(out, rows, cols, TILE_M, TILE_N);
    return result;
}
