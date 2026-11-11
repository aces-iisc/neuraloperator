// Number formats the matrix unit accepts, the tile layouts its unpackers expect, and the
// float64 reference the results are judged against.
#pragma once

#include <cstdint>
#include <vector>

// -- formats ---------------------------------------------------------------------------------
// The matrix unit has no fp32 multiply.  Operands arrive as bf16 or tf32 and Dst accumulates in
// fp32, so the widest thing the multipliers ever see is a tf32 mantissa -- narrower still once
// the fidelity phases take their slice.  Converting on the host keeps the comparison honest:
// the reference sees the same rounded operands the hardware does.
uint16_t to_bf16(float value);
float from_bf16(uint16_t bits);

// tf32 keeps fp32's exponent and rounds the mantissa to 10 bits.  It occupies four bytes in L1.
float to_tf32(float value);

// -- tile layouts ----------------------------------------------------------------------------
// A matrix is handed to the hardware as a grid of fixed-size tiles, each tile row-major and
// contiguous, because that is what one unpack moves.  Rows and columns past the edge are zero,
// which is why a band narrower than a tile costs the same as a full tile: padding is the
// hardware's unit of work, so the sweeps below step in powers of two that land on it.
//
// `rows` x `cols` is the logical matrix; `tile_rows` x `tile_cols` the tile.  Returns the bytes
// to write to L1, in tile-row-major order.
std::vector<uint8_t> pack_tiles(const std::vector<float> &matrix, uint32_t rows, uint32_t cols,
                                uint32_t tile_rows, uint32_t tile_cols, uint32_t format);

// The inverse, for fp32 output tiles coming back from the packer.
std::vector<float> unpack_tiles_f32(const std::vector<uint8_t> &bytes, uint32_t rows, uint32_t cols,
                                    uint32_t tile_rows, uint32_t tile_cols);

inline uint32_t tiles_for(uint32_t extent, uint32_t tile) {
    return (extent + tile - 1) / tile;
}

// -- the reference ---------------------------------------------------------------------------
// Everything is judged against a float64 matmul of the same rounded operands, so what the error
// measures is the matrix unit's arithmetic and not the choice of input format.
std::vector<double> reference_matmul(const std::vector<float> &a, const std::vector<float> &b,
                                     uint32_t rows, uint32_t depth, uint32_t cols);

// Largest absolute difference over the largest reference magnitude, which is the same relative
// error the GPU and TPU sweeps report.
double relative_error(const std::vector<float> &got, const std::vector<double> &want);
