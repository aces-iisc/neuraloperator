#include "engines.h"
#include "numerics.h"
#include "../include/tensix.h"

#include <cmath>
#include <numbers>

namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

// cos and sin of the DFT restricted to `rows`, stacked along the output axis so that one matmul
// produces both halves.  Stacking rather than issuing two matmuls is the same choice the GPU and
// TPU ports make: cos and sin multiply the same operand, so the second matmul would re-read it.
std::vector<float> stacked_rows(const std::vector<uint32_t> &rows, uint32_t length) {
    std::vector<float> out(size_t(2) * rows.size() * length);
    const size_t band = rows.size();
    for (size_t i = 0; i < band; ++i) {
        for (uint32_t j = 0; j < length; ++j) {
            const double angle = -kTwoPi * double(rows[i]) * double(j) / double(length);
            out[i * length + j] = float(std::cos(angle));
            out[(band + i) * length + j] = float(std::sin(angle));
        }
    }
    return out;
}

std::vector<uint32_t> band_rows(uint32_t n_modes) {
    std::vector<uint32_t> rows(n_modes);
    for (uint32_t k = 0; k < n_modes; ++k) {
        rows[k] = k;
    }
    return rows;
}

// The band's own rows followed by each bin's mirror.  Recovering two packed spectra needs both
// Z[k] and Z[(n - k) mod n], so a band of m modes forces 2m-1 rows rather than m.  That is the
// coupled scheme's real cost, and narrowing the band does not reduce it.
std::vector<uint32_t> mirrored_band_rows(uint32_t n, uint32_t n_modes) {
    std::vector<uint32_t> rows = band_rows(n_modes);
    for (uint32_t k = 1; k < n_modes; ++k) {
        rows.push_back((n - k) % n);
    }
    return rows;
}

Spectrum fail(const char *message) {
    Spectrum out;
    out.error = message;
    return out;
}

Spectrum run_reim(Device &device, const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                  uint32_t n_modes, uint32_t phases, uint32_t format) {
    const std::vector<float> matrix = stacked_rows(band_rows(n_modes), n);
    const MatmulResult got = tile_matmul(device, matrix, signal, 2 * n_modes, n, k_cols, phases,
                                         format);
    Spectrum out;
    out.mvmuls = got.mvmuls;
    out.unpacks = got.unpacks;
    out.packs = got.packs;
    out.clocks = got.clocks;
    out.timed_out = got.timed_out;
    if (got.timed_out) {
        return out;
    }
    const size_t half = size_t(n_modes) * k_cols;
    out.re.assign(got.y.begin(), got.y.begin() + half);
    out.im.assign(got.y.begin() + half, got.y.begin() + 2 * half);
    return out;
}

Spectrum run_coupled(Device &device, const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                     uint32_t n_modes, uint32_t phases, uint32_t format) {
    if (k_cols % 2) {
        return fail("coupled needs an even number of signals");
    }
    const uint32_t cols = k_cols / 2;
    const std::vector<uint32_t> rows = mirrored_band_rows(n, n_modes);
    const uint32_t n_rows = uint32_t(rows.size());

    // The two packed halves are the left and right column blocks of the input, so the operand is
    // the input itself and pairing the signals costs nothing to form.
    const std::vector<float> matrix = stacked_rows(rows, n);
    const MatmulResult got = tile_matmul(device, matrix, signal, 2 * n_rows, n, k_cols, phases,
                                         format);
    Spectrum out;
    out.mvmuls = got.mvmuls;
    out.unpacks = got.unpacks;
    out.packs = got.packs;
    out.clocks = got.clocks;
    out.timed_out = got.timed_out;
    if (got.timed_out) {
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

Spectrum run_reduced(Device &device, const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                     uint32_t n_modes, uint32_t phases, uint32_t format) {
    if (n % 2) {
        return fail("reduced needs an even signal length");
    }
    const uint32_t half = n / 2;
    if (n_modes > half) {
        return fail("reduced cannot reach a band wider than half the grid");
    }

    // An (n x k) array read as (n/2 x 2k) already has [even_i | odd_i] side by side in row i, so
    // the operand needs no stride-2 gather: it is the same bytes, re-shaped.
    const std::vector<float> matrix = stacked_rows(band_rows(n_modes), half);
    const MatmulResult got = tile_matmul(device, matrix, signal, 2 * n_modes, half, 2 * k_cols,
                                         phases, format);
    Spectrum out;
    out.mvmuls = got.mvmuls;
    out.unpacks = got.unpacks;
    out.packs = got.packs;
    out.clocks = got.clocks;
    out.timed_out = got.timed_out;
    if (got.timed_out) {
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

}  // namespace

const char *engine_name(Engine engine) {
    switch (engine) {
        case Engine::Reim: return "reim";
        case Engine::Coupled: return "coupled";
        case Engine::Reduced: return "reduced";
    }
    return "?";
}

Spectrum transform(Device &device, Engine engine, const std::vector<float> &signal, uint32_t n,
                   uint32_t k_cols, uint32_t n_modes, uint32_t phases, uint32_t format) {
    if (n_modes > n) {
        return fail("a band wider than the grid does not exist");
    }
    switch (engine) {
        case Engine::Reim: return run_reim(device, signal, n, k_cols, n_modes, phases, format);
        case Engine::Coupled:
            return run_coupled(device, signal, n, k_cols, n_modes, phases, format);
        case Engine::Reduced:
            return run_reduced(device, signal, n, k_cols, n_modes, phases, format);
    }
    return fail("unknown engine");
}

uint64_t engine_mvmuls(Engine engine, uint32_t n, uint32_t k_cols, uint32_t n_modes,
                       uint32_t phases) {
    switch (engine) {
        case Engine::Reim:
            return expected_mvmuls(2 * n_modes, n, k_cols, phases);
        case Engine::Coupled:
            return expected_mvmuls(2 * (2 * n_modes - 1), n, k_cols, phases);
        case Engine::Reduced:
            return expected_mvmuls(2 * n_modes, n / 2, 2 * k_cols, phases);
    }
    return 0;
}

void exact_spectrum(const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                    uint32_t n_modes, std::vector<double> &re, std::vector<double> &im) {
    re.assign(size_t(n_modes) * k_cols, 0.0);
    im.assign(size_t(n_modes) * k_cols, 0.0);
    for (uint32_t k = 0; k < n_modes; ++k) {
        for (uint32_t j = 0; j < n; ++j) {
            const double angle = -kTwoPi * double(k) * double(j) / double(n);
            const double c = std::cos(angle), s = std::sin(angle);
            for (uint32_t col = 0; col < k_cols; ++col) {
                const double x = signal[size_t(j) * k_cols + col];
                re[size_t(k) * k_cols + col] += c * x;
                im[size_t(k) * k_cols + col] += s * x;
            }
        }
    }
}

double spectrum_error(const Spectrum &got, const std::vector<double> &re,
                      const std::vector<double> &im) {
    if (got.re.size() != re.size() || got.im.size() != im.size()) {
        return 1.0;
    }
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < re.size(); ++i) {
        worst = std::max(worst, std::abs(double(got.re[i]) - re[i]));
        worst = std::max(worst, std::abs(double(got.im[i]) - im[i]));
        scale = std::max(scale, std::max(std::abs(re[i]), std::abs(im[i])));
    }
    return scale > 0.0 ? worst / scale : worst;
}
