#include "numerics.h"
#include "../include/tensix.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

uint32_t bits_of(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float float_of(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Round a mantissa to `keep` bits, to nearest with ties to even, leaving the exponent alone.
[[maybe_unused]] uint32_t round_mantissa(uint32_t bits, uint32_t keep) {
    const uint32_t drop = 23 - keep;
    if (drop == 0) {
        return bits;
    }
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if (exponent == 0xFFu) {
        return bits;                      // inf and NaN pass through untouched
    }
    const uint32_t half = 1u << (drop - 1);
    const uint32_t mask = (1u << drop) - 1;
    const uint32_t remainder = bits & mask;
    uint32_t rounded = bits & ~mask;
    if (remainder > half || (remainder == half && ((bits >> drop) & 1u))) {
        rounded += 1u << drop;
    }
    return rounded;
}

}  // namespace

// The unpacker truncates on the way into SrcA and SrcB rather than rounding, and flushes
// denormals, so the host copy is narrowed the same way.  Rounding here instead would make the
// reference disagree with the hardware by half a bit everywhere and hide what is being measured.
uint16_t to_bf16(float value) {
    const uint32_t bits = bits_of(value);
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    return uint16_t((exponent == 0 ? (bits & 0x80000000u) : bits) >> 16);
}

float from_bf16(uint16_t bits) {
    return float_of(uint32_t(bits) << 16);
}

float to_tf32(float value) {
    return float_of(bits_of(value) & 0xFFFFE000u);
}

std::vector<uint8_t> pack_tiles(const std::vector<float> &matrix, uint32_t rows, uint32_t cols,
                                uint32_t tile_rows, uint32_t tile_cols, uint32_t format) {
    if (matrix.size() != size_t(rows) * cols) {
        throw std::runtime_error("pack_tiles: matrix size does not match its shape");
    }
    const uint32_t row_tiles = tiles_for(rows, tile_rows);
    const uint32_t col_tiles = tiles_for(cols, tile_cols);
    const uint32_t bytes = format == FMT_BF16 ? 2u : 4u;
    std::vector<uint8_t> out(size_t(row_tiles) * col_tiles * tile_rows * tile_cols * bytes, 0);

    size_t at = 0;
    for (uint32_t ti = 0; ti < row_tiles; ++ti) {
        for (uint32_t tj = 0; tj < col_tiles; ++tj) {
            for (uint32_t r = 0; r < tile_rows; ++r) {
                for (uint32_t c = 0; c < tile_cols; ++c, at += bytes) {
                    const uint32_t row = ti * tile_rows + r;
                    const uint32_t col = tj * tile_cols + c;
                    if (row >= rows || col >= cols) {
                        continue;                      // the pad stays zero
                    }
                    const float value = matrix[size_t(row) * cols + col];
                    if (format == FMT_BF16) {
                        const uint16_t narrow = to_bf16(value);
                        std::memcpy(&out[at], &narrow, sizeof(narrow));
                    } else {
                        const float narrow = format == FMT_TF32 ? to_tf32(value) : value;
                        std::memcpy(&out[at], &narrow, sizeof(narrow));
                    }
                }
            }
        }
    }
    return out;
}

std::vector<float> unpack_tiles_f32(const std::vector<uint8_t> &bytes, uint32_t rows, uint32_t cols,
                                    uint32_t tile_rows, uint32_t tile_cols) {
    const uint32_t row_tiles = tiles_for(rows, tile_rows);
    const uint32_t col_tiles = tiles_for(cols, tile_cols);
    std::vector<float> out(size_t(rows) * cols, 0.0f);

    size_t at = 0;
    for (uint32_t ti = 0; ti < row_tiles; ++ti) {
        for (uint32_t tj = 0; tj < col_tiles; ++tj) {
            for (uint32_t r = 0; r < tile_rows; ++r) {
                for (uint32_t c = 0; c < tile_cols; ++c, at += 4) {
                    const uint32_t row = ti * tile_rows + r;
                    const uint32_t col = tj * tile_cols + c;
                    if (row >= rows || col >= cols || at + 4 > bytes.size()) {
                        continue;
                    }
                    float value;
                    std::memcpy(&value, &bytes[at], sizeof(value));
                    out[size_t(row) * cols + col] = value;
                }
            }
        }
    }
    return out;
}

std::vector<double> reference_matmul(const std::vector<float> &a, const std::vector<float> &b,
                                     uint32_t rows, uint32_t depth, uint32_t cols) {
    std::vector<double> out(size_t(rows) * cols, 0.0);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t k = 0; k < depth; ++k) {
            const double left = a[size_t(i) * depth + k];
            if (left == 0.0) {
                continue;
            }
            for (uint32_t j = 0; j < cols; ++j) {
                out[size_t(i) * cols + j] += left * double(b[size_t(k) * cols + j]);
            }
        }
    }
    return out;
}

double relative_error(const std::vector<float> &got, const std::vector<double> &want) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        worst = std::max(worst, std::abs(double(got[i]) - want[i]));
        scale = std::max(scale, std::abs(want[i]));
    }
    return scale > 0.0 ? worst / scale : worst;
}
