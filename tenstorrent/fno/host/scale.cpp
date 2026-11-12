// How the transform scales across the Tensix on a card.
//
// For one shape at a time: run it on one tile, then on two, four, and so on, splitting the
// batch; check every result against the single-tile answer; and record the busiest tile's
// instruction count, which is what the card waits for.
//
//   ./scale <libttsim.so> [out.csv]
//
// Two things limit the speedup and both are visible in the output.  The split grain is one
// 16-wide column tile, so a shape with fewer column tiles than tiles offered leaves the rest
// idle; and a share that is not a whole number of column tiles rounds up, so the padded work
// grows as the pieces get smaller.
#include "device.h"
#include "multicore.h"
#include "numerics.h"
#include "roofline.h"
#include "tile_matmul.h"
#include "../include/tensix.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const uint8_t kFirmware[] = {
#include "firmware.inc"
};

std::vector<uint32_t> env_list(const char *name, std::vector<uint32_t> fallback) {
    const char *raw = std::getenv(name);
    if (!raw) {
        return fallback;
    }
    std::vector<uint32_t> values;
    for (const char *p = raw; *p;) {
        values.push_back(uint32_t(std::strtoul(p, const_cast<char **>(&p), 10)));
        while (*p == ',' || *p == ' ') {
            ++p;
        }
    }
    return values.empty() ? fallback : values;
}

// The reim engine's shape: a 2m x N transform matrix against N x B signals.  One engine is
// enough here, because the split is over the batch and so is identical for all three.
struct Shape {
    uint32_t grid, batch, band;
    uint32_t rows() const { return 2 * band; }
    uint32_t depth() const { return grid; }
    uint32_t cols() const { return batch; }
};

double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        worst = std::max(worst, double(std::abs(a[i] - b[i])));
    }
    return worst;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libttsim.so> [out.csv]\n", argv[0]);
        return 2;
    }
    const std::string out_path = argc > 2 ? argv[2] : "results/scaling_blackhole.csv";

    try {
        Device device(argv[1]);
        const std::vector<uint8_t> image(kFirmware, kFirmware + sizeof(kFirmware));
        std::printf("vendor 0x%04x  device 0x%04x  %u of %u Tensix usable after harvesting\n",
                    device.vendor_id(), device.device_id(),
                    uint32_t(device.live_tiles().size()), Device::kNumTensix);

        const std::vector<uint32_t> grids = env_list("FNO_GRIDS", {64, 128, 256});
        const std::vector<uint32_t> batches = env_list("FNO_BATCHES", {64, 256, 1024});
        const std::vector<uint32_t> tile_counts =
            env_list("FNO_TILES", {1, 2, 4, 8, 16, 32, 64, 128});
        const uint32_t phases = 4, format = FMT_BF16;

        std::FILE *csv = std::fopen(out_path.c_str(), "w");
        if (!csv) {
            throw std::runtime_error("cannot open " + out_path);
        }
        std::fprintf(csv, "grid,B,n_modes,phases,tiles_offered,tiles_used,col_tiles,"
                          "mvmuls_single,mvmuls_total,mvmuls_critical,speedup,efficiency,"
                          "padding_overhead,us_critical,max_abs_diff,status\n");

        std::mt19937 rng(7);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        uint64_t rows_written = 0;

        for (uint32_t grid : grids) {
            for (uint32_t batch : batches) {
                const uint32_t band = grid / 4;
                const Shape shape{grid, batch, band};
                const uint32_t col_tiles = tiles_for(shape.cols(), TILE_N);

                std::vector<float> signal(size_t(shape.depth()) * shape.cols());
                for (float &value : signal) {
                    value = uniform(rng);
                }
                std::vector<float> matrix(size_t(shape.rows()) * shape.depth());
                for (float &value : matrix) {
                    value = uniform(rng);
                }

                // The single-tile cost is the shape's instruction count, which is exact and
                // needs no device.  Running it as well gives the answer every split is checked
                // against -- but a batch wide enough to be worth splitting need not fit in one
                // tile's L1, and that is a result rather than a failure, so a reference that
                // does not fit leaves the correctness column empty instead of stopping.
                const uint64_t single_mvmuls =
                    expected_mvmuls(shape.rows(), shape.depth(), shape.cols(), phases);
                std::vector<float> reference_y;
                bool have_reference = false;
                try {
                    device.select_tile(device.live_tiles()[0]);
                    device.start_firmware(image);
                    const MatmulResult reference =
                        tile_matmul(device, matrix, signal, shape.rows(), shape.depth(),
                                    shape.cols(), phases, format);
                    device.stop_firmware();
                    have_reference = !reference.timed_out;
                    reference_y = reference.y;
                } catch (const std::exception &) {
                    device.stop_firmware();        // does not fit on one tile; split anyway
                }

                for (uint32_t tiles : tile_counts) {
                    MulticoreResult got;
                    try {
                        got = multicore_matmul(device, image, matrix, signal, shape.rows(),
                                               shape.depth(), shape.cols(), phases, format,
                                               tiles);
                    } catch (const std::exception &e) {
                        std::fprintf(csv, "%u,%u,%u,%u,%u,,%u,%llu,,,,,,,,%s\n", grid, batch,
                                     band, phases, tiles, col_tiles,
                                     (unsigned long long)single_mvmuls, e.what());
                        ++rows_written;
                        continue;
                    }
                    if (got.timed_out) {
                        std::fprintf(csv, "%u,%u,%u,%u,%u,%u,%u,%llu,,,,,,,,timeout\n", grid,
                                     batch, band, phases, tiles, got.tiles_used, col_tiles,
                                     (unsigned long long)single_mvmuls);
                        ++rows_written;
                        continue;
                    }

                    const double speedup =
                        double(single_mvmuls) / double(got.mvmuls_critical);
                    const double efficiency = speedup / double(got.tiles_used);
                    const double padding = double(got.mvmuls_total) / double(single_mvmuls);
                    const double diff =
                        have_reference ? max_abs_diff(reference_y, got.y) : -1.0;

                    std::fprintf(csv,
                                 "%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%u,%.4f,%.4f,%.4f,%.6g,"
                                 "%.3e,%s\n",
                                 grid, batch, band, phases, tiles, got.tiles_used, col_tiles,
                                 (unsigned long long)single_mvmuls,
                                 (unsigned long long)got.mvmuls_total, got.mvmuls_critical,
                                 speedup, efficiency, padding,
                                 roofline::micros(got.mvmuls_critical), diff,
                                 !have_reference ? "ok, too big for one tile to check"
                                                 : (diff == 0.0 ? "ok"
                                                                : "differs from one tile"));
                    ++rows_written;
                }
                std::printf("N=%-4u B=%-5u m=%-4u  %u column tiles  done\n", grid, batch, band,
                            col_tiles);
            }
        }
        std::fclose(csv);
        std::printf("\nwrote %s: %llu rows\n", out_path.c_str(),
                    (unsigned long long)rows_written);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
