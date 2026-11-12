#include "tricks.h"

#include <cmath>
#include <numbers>

namespace fft {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

}  // namespace

Spectrum rfft_coupled_low_modes(Device &device, const std::vector<float> &signal, uint32_t n,
                                uint32_t k_cols, uint32_t n_modes, uint32_t phases,
                                uint32_t format) {
    if (n_modes > n) {
        return fail("a band wider than the grid does not exist");
    }
    if (k_cols % 2) {
        return fail("coupled needs an even number of signals");
    }
    const uint32_t cols = k_cols / 2;
    const std::vector<uint32_t> rows = mirrored_band_rows(n, n_modes);
    const uint32_t n_rows = uint32_t(rows.size());

    // The two packed halves are the left and right column blocks of the input, so the operand is
    // the input itself and pairing the signals costs nothing to form.
    const std::vector<float> matrix = stacked_rows(rows, n);
    const MatmulResult got =
        tile_matmul(device, matrix, signal, 2 * n_rows, n, k_cols, phases, format);

    Spectrum out;
    if (!accumulate(out, got)) {
        return out;
    }

    const auto at = [&](uint32_t row, uint32_t col) { return got.y[size_t(row) * k_cols + col]; };
    out.re.assign(size_t(n_modes) * k_cols, 0.0f);
    out.im.assign(size_t(n_modes) * k_cols, 0.0f);

    for (uint32_t k = 0; k < n_modes; ++k) {
        // The mirror of bin k is row 0 for the DC bin and row n_modes + k - 1 otherwise, which
        // is the order mirrored_band_rows laid them in.
        const uint32_t mirror = k == 0 ? 0 : n_modes + k - 1;
        for (uint32_t c = 0; c < cols; ++c) {
            // (cos + i sin) @ (a + i b), with a and b the two column blocks.
            const float z_re = at(k, c) - at(n_rows + k, cols + c);
            const float z_im = at(n_rows + k, c) + at(k, cols + c);
            const float m_re = at(mirror, c) - at(n_rows + mirror, cols + c);
            const float m_im = at(n_rows + mirror, c) + at(mirror, cols + c);

            const size_t a_at = size_t(k) * k_cols + c;
            const size_t b_at = a_at + cols;
            out.re[a_at] = 0.5f * (z_re + m_re);
            out.im[a_at] = 0.5f * (z_im - m_im);
            out.re[b_at] = 0.5f * (z_im + m_im);
            out.im[b_at] = -0.5f * (z_re - m_re);
        }
    }
    return out;
}

Spectrum rfft_reduced_low_modes(Device &device, const std::vector<float> &signal, uint32_t n,
                                uint32_t k_cols, uint32_t n_modes, uint32_t phases,
                                uint32_t format) {
    if (n_modes > n) {
        return fail("a band wider than the grid does not exist");
    }
    if (n % 2) {
        return fail("reduced needs an even signal length");
    }
    const uint32_t half = n / 2;
    if (n_modes > half) {
        return fail("reduced cannot reach a band wider than half the grid");
    }

    const std::vector<float> matrix = stacked_rows(band_rows(n_modes), half);
    const MatmulResult got =
        tile_matmul(device, matrix, signal, 2 * n_modes, half, 2 * k_cols, phases, format);

    Spectrum out;
    if (!accumulate(out, got)) {
        return out;
    }

    const uint32_t wide = 2 * k_cols;
    const auto at = [&](uint32_t row, uint32_t col) { return got.y[size_t(row) * wide + col]; };
    out.re.assign(size_t(n_modes) * k_cols, 0.0f);
    out.im.assign(size_t(n_modes) * k_cols, 0.0f);

    for (uint32_t k = 0; k < n_modes; ++k) {
        const double angle = -kTwoPi * double(k) / double(n);
        const float w_re = float(std::cos(angle)), w_im = float(std::sin(angle));
        for (uint32_t c = 0; c < k_cols; ++c) {
            const float e_re = at(k, c), o_re = at(k, k_cols + c);
            const float e_im = at(n_modes + k, c), o_im = at(n_modes + k, k_cols + c);
            const size_t to = size_t(k) * k_cols + c;
            out.re[to] = e_re + w_re * o_re - w_im * o_im;   // X[k] = E[k] + W^k O[k]
            out.im[to] = e_im + w_re * o_im + w_im * o_re;
        }
    }
    return out;
}

}  // namespace fft
