// The three-stage firmware against the one-core firmware, shape by shape.
//
//   ./check_pipeline <libttsim.so>
//
// The claim being tested is narrow and deliberately so: splitting the work across three Tensix
// threads changes who issues each instruction and when, and changes nothing about the arithmetic.
// So the output must match the one-core answer *exactly*, not approximately, and the instruction
// counts must match too -- an overlap that silently dropped or repeated a reduction step would
// show up as a different MVMUL count even if the numbers happened to look plausible.
//
// What this cannot test is the overlap itself.  The simulator retires a tile's instructions
// eagerly, ignores STALLWAIT's block mask outright, and treats most of its conditions as already
// satisfied, so a stage boundary that is wrong on silicon can still produce the right answer
// here.  The semaphores are modelled and are exercised; the timing they protect is not.

#include "device.h"
#include "numerics.h"
#include "pipeline_matmul.h"
#include "tile_matmul.h"
#include "../include/tensix.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const uint8_t kFirmware[] = {
#include "firmware.inc"
};
const uint8_t kStage0[] = {
#include "stage0.inc"
};
const uint8_t kStage1[] = {
#include "stage1.inc"
};
const uint8_t kStage2[] = {
#include "stage2.inc"
};

struct Case {
    const char *name;
    uint32_t rows, depth, cols, phases, format;
};

// Shapes chosen to exercise each loop bound on its own: one tile in every direction, then more
// than one of each in turn, then the band-shaped matmul the transform actually issues, then the
// fidelity axis.  A stage boundary that mishandles the k loop fails the deep cases; one that
// mishandles the block loop fails the wide ones.
const Case kCases[] = {
    {"one tile each way", 8, 16, 16, 1, FMT_BF16},
    {"two reduction steps", 8, 32, 16, 1, FMT_BF16},
    {"four reduction steps", 8, 64, 16, 1, FMT_BF16},
    {"two output columns", 8, 16, 32, 1, FMT_BF16},
    {"two output rows", 16, 16, 16, 1, FMT_BF16},
    {"two rows by two columns", 16, 16, 32, 1, FMT_BF16},
    {"a band-shaped matmul", 32, 64, 32, 1, FMT_BF16},
    {"two phases", 16, 32, 32, 2, FMT_BF16},
    {"four phases", 16, 32, 32, 4, FMT_BF16},
    {"four phases, deep", 32, 128, 16, 4, FMT_BF16},
    {"tf32 operands", 16, 32, 32, 4, FMT_TF32},
    {"the widest that fits", 64, 64, 128, 4, FMT_BF16},
};

size_t count_differences(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        return a.size() > b.size() ? a.size() : b.size();
    }
    size_t different = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        // Bit comparison, not a tolerance: the two paths issue the same multiplies in the same
        // order, so anything but an identical result is a real difference.
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            ++different;
        }
    }
    return different;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libttsim.so>\n", argv[0]);
        return 2;
    }

    try {
        Device device(argv[1]);
        std::printf("vendor 0x%04x  device 0x%04x\n", device.vendor_id(), device.device_id());
        std::printf("one-core firmware %zu bytes; stages %zu / %zu / %zu bytes\n\n",
                    sizeof(kFirmware), sizeof(kStage0), sizeof(kStage1), sizeof(kStage2));

        const std::vector<uint8_t> baseline(kFirmware, kFirmware + sizeof(kFirmware));
        const std::vector<uint8_t> stage0(kStage0, kStage0 + sizeof(kStage0));
        const std::vector<uint8_t> stage1(kStage1, kStage1 + sizeof(kStage1));
        const std::vector<uint8_t> stage2(kStage2, kStage2 + sizeof(kStage2));
        const Device::StageImages stages{&stage0, &stage1, &stage2};

        std::mt19937 gen(11);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        unsigned failures = 0;

        for (const Case &test : kCases) {
            std::vector<float> t(size_t(test.rows) * test.depth);
            std::vector<float> x(size_t(test.depth) * test.cols);
            for (float &value : t) {
                value = dist(gen);
            }
            for (float &value : x) {
                value = dist(gen);
            }

            device.start_firmware(baseline);
            const MatmulResult one = tile_matmul(device, t, x, test.rows, test.depth, test.cols,
                                                 test.phases, test.format);
            device.stop_firmware();

            device.start_pipeline(stages);
            const PipelineResult three =
                pipeline_matmul(device, t, x, test.rows, test.depth, test.cols, test.phases,
                                test.format);
            device.stop_firmware();

            std::printf("  %-24s %4ux%-4ux%-4u p%u  ", test.name, test.rows, test.depth,
                        test.cols, test.phases);

            if (one.timed_out) {
                std::printf("one-core run timed out\n");
                ++failures;
                continue;
            }
            if (three.timed_out) {
                std::printf("stages timed out, done_mask 0x%x\n", three.done_mask);
                ++failures;
                continue;
            }

            const size_t different = count_differences(one.y, three.y);
            const bool counts_agree = one.mvmuls == three.mvmuls &&
                                      one.unpacks == three.unpacks && one.packs == three.packs;
            if (different == 0 && counts_agree) {
                std::printf("bit-exact   mvmul %u  unpack %u  pack %u   ok\n", three.mvmuls,
                            three.unpacks, three.packs);
                continue;
            }
            ++failures;
            if (different != 0) {
                std::printf("%zu of %zu values differ", different, one.y.size());
                if (std::getenv("FNO_BLOCKS")) {
                    // Classify each 8x16 output block: still poison means the packer never
                    // wrote it, "holds block n" means it was written with another block's
                    // result, which tells a missed pack apart from a misaddressed one.
                    const uint32_t width = tiles_for(test.cols, TILE_N) * TILE_N;
                    const uint32_t row_blocks = tiles_for(test.rows, TILE_M);
                    const uint32_t col_blocks = tiles_for(test.cols, TILE_N);
                    const float poison = []() {
                        uint32_t bits = 0xCDCDCDCDu;
                        float value;
                        std::memcpy(&value, &bits, sizeof(value));
                        return value;
                    }();
                    std::printf("\n");
                    for (uint32_t bi = 0; bi < row_blocks; ++bi) {
                        for (uint32_t bj = 0; bj < col_blocks; ++bj) {
                            const size_t head = size_t(bi) * TILE_M * width + bj * TILE_N;
                            const float got = three.y[head];
                            const char *verdict = "differs";
                            if (std::memcmp(&got, &poison, sizeof(float)) == 0) {
                                verdict = "never packed";
                            } else if (std::memcmp(&got, &one.y[head], sizeof(float)) == 0) {
                                verdict = "ok";
                            } else {
                                for (uint32_t oi = 0; oi < row_blocks; ++oi) {
                                    for (uint32_t oj = 0; oj < col_blocks; ++oj) {
                                        const size_t other =
                                            size_t(oi) * TILE_M * width + oj * TILE_N;
                                        if (std::memcmp(&got, &one.y[other], sizeof(float)) == 0) {
                                            static char where[48];
                                            std::snprintf(where, sizeof(where),
                                                          "holds block (%u,%u)", oi, oj);
                                            verdict = where;
                                        }
                                    }
                                }
                            }
                            std::printf("      block (%u,%u): %s\n", bi, bj, verdict);
                        }
                    }
                    std::printf("   ");
                }
                if (std::getenv("FNO_DIFF")) {
                    const uint32_t width = tiles_for(test.cols, TILE_N) * TILE_N;
                    std::printf("\n");
                    for (size_t i = 0, shown = 0; i < one.y.size() && shown < 8; ++i) {
                        if (std::memcmp(&one.y[i], &three.y[i], sizeof(float)) == 0) {
                            continue;
                        }
                        std::printf("      row %zu col %zu: one-core % .6f  stages % .6f\n",
                                    i / width, i % width, one.y[i], three.y[i]);
                        ++shown;
                    }
                    std::printf("   ");
                }
            }
            if (!counts_agree) {
                std::printf("%scounts one-core %u/%u/%u vs stages %u/%u/%u",
                            different != 0 ? "; " : "", one.mvmuls, one.unpacks, one.packs,
                            three.mvmuls, three.unpacks, three.packs);
            }
            std::printf("\n");
        }

        std::printf("\n%s\n", failures == 0 ? "all cases ok"
                                            : "cases differ from the one-core firmware");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
