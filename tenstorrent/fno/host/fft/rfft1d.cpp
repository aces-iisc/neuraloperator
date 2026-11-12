#include "rfft1d.h"

namespace fft {

Spectrum rfft_rows(Device &device, const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                   const std::vector<uint32_t> &rows, uint32_t phases, uint32_t format) {
    const uint32_t band = uint32_t(rows.size());
    if (band == 0) {
        return fail("an empty row set has no transform");
    }
    const std::vector<float> matrix = stacked_rows(rows, n);
    const MatmulResult got =
        tile_matmul(device, matrix, signal, 2 * band, n, k_cols, phases, format);

    Spectrum out;
    if (!accumulate(out, got)) {
        return out;
    }
    const size_t half = size_t(band) * k_cols;
    out.re.assign(got.y.begin(), got.y.begin() + half);
    out.im.assign(got.y.begin() + half, got.y.begin() + 2 * half);
    return out;
}

Spectrum rfft_low_modes(Device &device, const std::vector<float> &signal, uint32_t n,
                        uint32_t k_cols, uint32_t n_modes, uint32_t phases, uint32_t format) {
    if (n_modes > n) {
        return fail("a band wider than the grid does not exist");
    }
    return rfft_rows(device, signal, n, k_cols, band_rows(n_modes), phases, format);
}

Spectrum rfft_direct(Device &device, const std::vector<float> &signal, uint32_t n,
                     uint32_t k_cols, bool half, uint32_t phases, uint32_t format) {
    return rfft_rows(device, signal, n, k_cols, band_rows(half ? n / 2 : n), phases, format);
}

}  // namespace fft
