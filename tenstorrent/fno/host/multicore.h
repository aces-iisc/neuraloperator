// The same matmul, split over several Tensix tiles.
//
// A card is 120 or 140 Tensix, and the single-tile roofline in roofline.h assumes the work
// divides over them perfectly.  This measures whether it does, by actually running it: the
// signals are dealt out to tiles, every tile is started, and the simulator clocks all of them
// together.  What comes back is the busiest tile's instruction count, which is the parallel
// critical path and the honest denominator for a per-card projection.
//
// The split is over columns of X, which is the batch: every tile holds the whole transform
// matrix T and its own slice of the signals.  That needs no tile-to-tile traffic, which is
// what makes it the right first measurement -- any slower result is then the partitioning's
// fault and not the network's.
//
// The grain is one 16-wide column tile, because the matrix unit cannot take less.  So a shape
// with fewer column tiles than tiles offered leaves the surplus idle, and that ceiling is the
// first thing the scaling curve shows.
#pragma once

#include "device.h"

#include <cstdint>
#include <vector>

struct MulticoreResult {
    std::vector<float> y;            // rows x cols, gathered from every tile
    uint32_t tiles_offered = 0;      // how many this call was allowed to use
    uint32_t tiles_used = 0;         // how many got at least one column tile
    uint64_t mvmuls_total = 0;       // summed over the tiles that ran
    uint32_t mvmuls_critical = 0;    // the busiest tile: what the card actually waits for
    uint64_t clocks = 0;
    bool timed_out = false;

    // Work the same shape costs on one tile.  The ratio to mvmuls_critical is the speedup, and
    // the ratio to mvmuls_total is what the split wasted in padding.
    uint64_t mvmuls_single = 0;
};

// `t` is rows x depth, `x` is depth x cols, both row-major, as in tile_matmul.  Every tile is
// loaded with `image` and left stopped afterwards.
MulticoreResult multicore_matmul(Device &device, const std::vector<uint8_t> &image,
                                 const std::vector<float> &t, const std::vector<float> &x,
                                 uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases,
                                 uint32_t format, uint32_t tiles);
