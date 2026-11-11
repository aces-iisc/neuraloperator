// Is the matrix unit computing at all?
//
// The packer sits between Dst and L1, so an all-zero result in L1 has two causes that look
// identical from outside: the multipliers produced nothing, or they produced something the
// packer did not deliver.  This reads Dst through the tile's debug array port, which bypasses
// the packer entirely, and so separates the two.

#include "device.h"
#include "numerics.h"
#include "tile_matmul.h"
#include "../include/tensix.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
const uint8_t kFirmware[] = {
#include "firmware.inc"
};
}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libttsim.so>\n", argv[0]);
        return 2;
    }
    Device device(argv[1]);
    device.start_firmware(std::vector<uint8_t>(kFirmware, kFirmware + sizeof(kFirmware)));
    device.clock(200000);

    // Give every row of T a distinct value and make X's first column all ones.  Then output
    // row r must be 16*(r+1), so the answer names which row of T reached which row of SrcB and
    // which Dst row it landed in.  A permutation or a stride shows up as a readable pattern
    // rather than as a wrong number.
    std::vector<float> t(TILE_M * TILE_K, 0.0f), x(TILE_K * TILE_N, 0.0f);
    for (uint32_t r = 0; r < TILE_M; ++r) {
        for (uint32_t k = 0; k < TILE_K; ++k) {
            t[r * TILE_K + k] = float(r + 1);
        }
    }
    for (uint32_t k = 0; k < TILE_K; ++k) {
        x[k * TILE_N] = 1.0f;
    }

    const MatmulResult got = tile_matmul(device, t, x, TILE_M, TILE_K, TILE_N, 1, FMT_BF16);
    std::printf("mvmuls=%u timed_out=%d\n\n", got.mvmuls, int(got.timed_out));

    std::printf("packed into L1, column 0 of each output row (want 16, 32, ... 128):\n ");
    for (uint32_t r = 0; r < TILE_M; ++r) {
        std::printf("%10.2f", got.y[size_t(r) * TILE_N]);
    }
    std::printf("\n\nDst as stored, column 0 of rows 0..15 (high halves then low):\n ");
    for (uint32_t row = 0; row < 16; ++row) {
        std::printf(" %04x", device.read_dst_raw(row, 0) & 0xFFFFu);
    }
    std::printf("\n\nthe same, decoded as fp32 (high row r paired with low row r+8):\n ");
    for (uint32_t r = 0; r < TILE_M; ++r) {
        // Dst keeps fp32 as two halves eight rows apart, and each half stores the exponent and
        // mantissa swapped relative to IEEE.  Undo both to read it as a number.
        const uint32_t high = device.read_dst_raw(r, 0) & 0xFFFFu;
        const uint32_t low = device.read_dst_raw(r + 8, 0) & 0xFFFFu;
        const uint32_t ieee_high = (high & 0x8000u) | (((high >> 0) & 0xFFu) << 7) | ((high >> 8) & 0x7Fu);
        float value;
        const uint32_t bits = (ieee_high << 16) | low;
        std::memcpy(&value, &bits, sizeof(value));
        std::printf("%10.2f", value);
    }
    std::printf("\n");

    device.stop_firmware();
    return 0;
}
