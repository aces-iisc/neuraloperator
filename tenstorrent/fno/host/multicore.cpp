#include "multicore.h"
#include "numerics.h"
#include "tile_matmul.h"
#include "../include/tensix.h"

#include <algorithm>
#include <stdexcept>

namespace {

constexpr uint64_t kClockBudget = 200ull * 1000 * 1000;
constexpr uint32_t kClockChunk = 2000u;

uint32_t command_field(uint32_t index) {
    return L1_COMMAND_ADDR + index * 4u;
}

// One tile's share of the column tiles.  The remainder is spread one apiece over the first
// tiles rather than piled onto the last, so the critical path is the smallest it can be for
// this grain.
struct Share {
    uint32_t first_col_tile;
    uint32_t col_tiles;
};

std::vector<Share> deal(uint32_t col_tiles, uint32_t tiles) {
    std::vector<Share> shares;
    uint32_t next = 0;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t remaining_tiles = tiles - tile;
        const uint32_t take = (col_tiles - next + remaining_tiles - 1) / remaining_tiles;
        if (take == 0) {
            break;                                  // nothing left; the rest stay idle
        }
        shares.push_back({next, take});
        next += take;
    }
    return shares;
}

// The columns of `x` that one share covers, as a dense depth x width matrix.  The tail share of
// a shape whose width is not a multiple of the tile is zero-filled, exactly as the single-tile
// path pads.
std::vector<float> slice_columns(const std::vector<float> &x, uint32_t depth, uint32_t cols,
                                 uint32_t first_col, uint32_t width) {
    std::vector<float> out(size_t(depth) * width, 0.0f);
    for (uint32_t row = 0; row < depth; ++row) {
        for (uint32_t column = 0; column < width; ++column) {
            const uint32_t source = first_col + column;
            if (source < cols) {
                out[size_t(row) * width + column] = x[size_t(row) * cols + source];
            }
        }
    }
    return out;
}

}  // namespace

MulticoreResult multicore_matmul(Device &device, const std::vector<uint8_t> &image,
                                 const std::vector<float> &t, const std::vector<float> &x,
                                 uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases,
                                 uint32_t format, uint32_t tiles) {
    if (phases < 1 || phases > 4) {
        throw std::runtime_error("fidelity phases must be 1..4");
    }
    const std::vector<uint32_t> &live = device.live_tiles();
    if (tiles == 0) {
        throw std::runtime_error("tile count out of range");
    }
    tiles = std::min(tiles, uint32_t(live.size()));
    const uint32_t row_tiles = tiles_for(rows, TILE_M);
    const uint32_t depth_tiles = tiles_for(depth, TILE_K);
    const uint32_t col_tiles = tiles_for(cols, TILE_N);

    MulticoreResult result;
    result.tiles_offered = tiles;
    result.mvmuls_single = expected_mvmuls(rows, depth, cols, phases);

    const std::vector<Share> shares = deal(col_tiles, tiles);
    result.tiles_used = uint32_t(shares.size());

    const std::vector<uint8_t> t_bytes = pack_tiles(t, rows, depth, TILE_M, TILE_K, FMT_FP32);

    // Load and start every tile before clocking any of them, so they run together rather than
    // one after another.
    for (uint32_t index = 0; index < shares.size(); ++index) {
        const Share &share = shares[index];
        const uint32_t width = share.col_tiles * TILE_N;
        const std::vector<float> x_slice =
            slice_columns(x, depth, cols, share.first_col_tile * TILE_N, width);
        const std::vector<uint8_t> x_bytes =
            pack_tiles(x_slice, depth, width, TILE_K, TILE_N, FMT_FP32);
        const size_t y_bytes = size_t(row_tiles) * share.col_tiles * TILE_M * TILE_N * 4;

        if (L1_MATRIX_ADDR + t_bytes.size() > L1_OPERAND_ADDR ||
            L1_OPERAND_ADDR + x_bytes.size() > L1_OUTPUT_ADDR ||
            L1_OUTPUT_ADDR + y_bytes > L1_SIZE) {
            throw std::runtime_error("shape does not fit in L1");
        }

        device.select_tile(live[index]);
        device.start_firmware(image);
        device.write_l1(L1_MATRIX_ADDR, t_bytes.data(), uint32_t(t_bytes.size()));
        device.write_l1(L1_OPERAND_ADDR, x_bytes.data(), uint32_t(x_bytes.size()));
        const std::vector<uint8_t> poison(y_bytes, 0xCD);
        device.write_l1(L1_OUTPUT_ADDR, poison.data(), uint32_t(poison.size()));

        device.write_l1_u32(command_field(2), row_tiles);
        device.write_l1_u32(command_field(3), depth_tiles);
        device.write_l1_u32(command_field(4), share.col_tiles);
        device.write_l1_u32(command_field(5), phases);
        device.write_l1_u32(command_field(6), format);
        device.write_l1_u32(command_field(1), CMD_IDLE);
        device.write_l1_u32(command_field(0), CMD_MATMUL);
    }

    // One clock advances every busy tile, so this waits on all of them at once.
    uint64_t spent = 0;
    bool all_done = false;
    while (spent < kClockBudget && !all_done) {
        device.clock(kClockChunk);
        spent += kClockChunk;
        all_done = true;
        for (uint32_t index = 0; index < shares.size(); ++index) {
            device.select_tile(live[index]);
            if (device.read_l1_u32(command_field(1)) != CMD_MATMUL) {
                all_done = false;
                break;
            }
        }
    }
    result.clocks = spent;
    if (!all_done) {
        result.timed_out = true;
        for (uint32_t index = 0; index < shares.size(); ++index) {
            device.select_tile(live[index]);
            device.stop_firmware();
        }
        return result;
    }

    result.y.assign(size_t(rows) * cols, 0.0f);
    for (uint32_t index = 0; index < shares.size(); ++index) {
        const Share &share = shares[index];
        const uint32_t width = share.col_tiles * TILE_N;
        device.select_tile(live[index]);

        const uint32_t mvmuls = device.read_l1_u32(command_field(7));
        result.mvmuls_total += mvmuls;
        result.mvmuls_critical = std::max(result.mvmuls_critical, mvmuls);

        std::vector<uint8_t> out(size_t(row_tiles) * share.col_tiles * TILE_M * TILE_N * 4);
        device.read_l1(L1_OUTPUT_ADDR, out.data(), uint32_t(out.size()));
        const std::vector<float> block = unpack_tiles_f32(out, rows, width, TILE_M, TILE_N);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < width; ++column) {
                const uint32_t target = share.first_col_tile * TILE_N + column;
                if (target < cols) {
                    result.y[size_t(row) * cols + target] = block[size_t(row) * width + column];
                }
            }
        }
        device.stop_firmware();
    }
    device.select_tile(live[0]);
    return result;
}
