// The unit everything else rests on: one matmul on the matrix unit, against a float64 reference.
//
// Nothing in this port is believable until this is, so it runs first and prints enough to tell
// the three ways it can fail apart: the tile never answers, the tile answers with the wrong
// numbers, or the tile answers correctly but issues the wrong number of instructions.

#include "device.h"
#include "numerics.h"
#include "tile_matmul.h"
#include "../include/tensix.h"

#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

const uint8_t kFirmware[] = {
#include "firmware.inc"
};

// Operands are rounded to the hardware's format before the reference sees them, so the error
// reported is the matrix unit's and not the input format's.
std::vector<float> rounded(const std::vector<float> &values, uint32_t format) {
    std::vector<float> out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = format == FMT_BF16 ? from_bf16(to_bf16(values[i])) : to_tf32(values[i]);
    }
    return out;
}

struct Case {
    const char *name;
    uint32_t rows, depth, cols, phases, format;
    double tolerance;   // what the operand format and the phase count can actually deliver
};

bool run(Device &device, const Case &c) {
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> t(size_t(c.rows) * c.depth), x(size_t(c.depth) * c.cols);
    for (auto &v : t) v = dist(gen);
    for (auto &v : x) v = dist(gen);

    const MatmulResult got = tile_matmul(device, t, x, c.rows, c.depth, c.cols, c.phases, c.format);
    if (got.timed_out) {
        std::printf("  %-28s TIMEOUT after the clock budget\n", c.name);
        return false;
    }

    const std::vector<double> want = reference_matmul(rounded(t, c.format), rounded(x, c.format),
                                                      c.rows, c.depth, c.cols);
    const double error = relative_error(got.y, want);
    const uint64_t expected = expected_mvmuls(c.rows, c.depth, c.cols, c.phases);
    const bool counted = got.mvmuls == expected;

    std::printf("  %-28s rel_err %9.3e   mvmul %5u/%-5llu %s   %llu clocks\n", c.name, error,
                got.mvmuls, (unsigned long long)expected, counted ? "ok " : "BAD",
                (unsigned long long)got.clocks);

    if (error >= c.tolerance) {
        std::printf("      got  ");
        for (uint32_t i = 0; i < 6 && i < got.y.size(); ++i) std::printf("%12.5f", got.y[i]);
        std::printf("\n      want ");
        for (uint32_t i = 0; i < 6 && i < want.size(); ++i) std::printf("%12.5f", want[i]);
        std::printf("\n");
    }
    return counted && error < c.tolerance;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libttsim.so>\n", argv[0]);
        return 2;
    }
    try {
        Device device(argv[1]);
        std::printf("vendor 0x%04x  device 0x%04x  firmware %zu bytes\n", device.vendor_id(),
                    device.device_id(), sizeof(kFirmware));

        device.start_firmware(std::vector<uint8_t>(kFirmware, kFirmware + sizeof(kFirmware)));

        // The core has to reach its polling loop before a command means anything.
        device.clock(200000);
        std::printf("after boot: done=%u heartbeat=%u\n", device.read_l1_u32(L1_COMMAND_ADDR + 4),
                    device.read_l1_u32(L1_COMMAND_ADDR + 32));

        // One phase sees five bits of SrcA against seven of SrcB, so a few percent is the
        // right answer there, not a failure.  Four phases spend four times the instructions to
        // buy the operand format's full mantissa back.
        const Case cases[] = {
            {"one tile, one phase",     8,  16,  16, 1, FMT_BF16, 5e-2},
            {"one tile, four phases",   8,  16,  16, 4, FMT_BF16, 1e-3},
            {"deep reduction",          8,  64,  16, 4, FMT_BF16, 1e-3},
            {"several output rows",    32,  16,  16, 4, FMT_BF16, 1e-3},
            {"several signal tiles",    8,  16,  64, 4, FMT_BF16, 1e-3},
            {"a band-shaped matmul",   32,  64,  64, 4, FMT_BF16, 1e-3},
            {"tf32 operands",          32,  64,  64, 4, FMT_TF32, 1e-3},
            {"two phases",             32,  64,  64, 2, FMT_BF16, 5e-2},
        };
        bool all = true;
        for (const Case &c : cases) {
            all = run(device, c) && all;
        }
        device.stop_firmware();
        std::printf("%s\n", all ? "all cases ok" : "FAILURES above");
        return all ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
