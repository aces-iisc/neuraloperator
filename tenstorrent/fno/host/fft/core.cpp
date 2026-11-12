#include "core.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace fft {
namespace {

constexpr double kTwoPi = 2.0 * std::numbers::pi;

}  // namespace

bool accumulate(Spectrum &out, const MatmulResult &got) {
    out.mvmuls += got.mvmuls;
    out.unpacks += got.unpacks;
    out.packs += got.packs;
    out.clocks += got.clocks;
    if (got.timed_out) {
        out.timed_out = true;
        return false;
    }
    return true;
}

Spectrum fail(const char *message) {
    Spectrum out;
    out.error = message;
    return out;
}

std::vector<uint32_t> band_rows(uint32_t n_modes) {
    std::vector<uint32_t> rows(n_modes);
    for (uint32_t k = 0; k < n_modes; ++k) {
        rows[k] = k;
    }
    return rows;
}

std::vector<uint32_t> mirrored_band_rows(uint32_t n, uint32_t n_modes) {
    std::vector<uint32_t> rows = band_rows(n_modes);
    for (uint32_t k = 1; k < n_modes; ++k) {
        rows.push_back((n - k) % n);
    }
    return rows;
}

void rows_matrix(const std::vector<uint32_t> &rows, uint32_t length, std::vector<float> &cos_out,
                 std::vector<float> &sin_out) {
    const size_t band = rows.size();
    cos_out.assign(band * length, 0.0f);
    sin_out.assign(band * length, 0.0f);
    for (size_t i = 0; i < band; ++i) {
        for (uint32_t j = 0; j < length; ++j) {
            const double angle = -kTwoPi * double(rows[i]) * double(j) / double(length);
            cos_out[i * length + j] = float(std::cos(angle));
            sin_out[i * length + j] = float(std::sin(angle));
        }
    }
}

std::vector<float> stacked_rows(const std::vector<uint32_t> &rows, uint32_t length) {
    std::vector<float> cos_rows, sin_rows;
    rows_matrix(rows, length, cos_rows, sin_rows);
    std::vector<float> out(cos_rows.size() + sin_rows.size());
    std::copy(cos_rows.begin(), cos_rows.end(), out.begin());
    std::copy(sin_rows.begin(), sin_rows.end(), out.begin() + cos_rows.size());
    return out;
}

ComplexProduct complex_matmul(
    const std::function<std::vector<float>(const std::vector<float> &, const std::vector<float> &)>
        &matmul,
    const std::vector<float> &cos_rows, const std::vector<float> &sin_rows,
    const std::vector<float> &cos_plus_sin_rows, const std::vector<float> &x_re,
    const std::vector<float> &x_im) {
    std::vector<float> sum(x_re.size());
    for (size_t i = 0; i < x_re.size(); ++i) {
        sum[i] = x_re[i] + x_im[i];
    }

    const std::vector<float> a = matmul(cos_rows, x_re);
    const std::vector<float> b = matmul(sin_rows, x_im);
    const std::vector<float> c = matmul(cos_plus_sin_rows, sum);

    ComplexProduct out;
    if (a.size() != b.size() || a.size() != c.size()) {
        return out;                                  // a matmul failed; the caller sees empty
    }
    out.re.resize(a.size());
    out.im.resize(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        out.re[i] = a[i] - b[i];
        out.im[i] = c[i] - a[i] - b[i];
    }
    return out;
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

void exact_spectrum2(const std::vector<float> &field, uint32_t n_rows, uint32_t n_cols,
                     uint32_t batch, uint32_t row_modes, uint32_t col_modes,
                     std::vector<double> &re, std::vector<double> &im) {
    // Pass one: the real row transform, keeping row_modes bins.  (row_modes x n_cols x batch),
    // complex from here on.
    std::vector<double> mid_re(size_t(row_modes) * n_cols * batch, 0.0);
    std::vector<double> mid_im(mid_re.size(), 0.0);
    for (uint32_t a = 0; a < row_modes; ++a) {
        for (uint32_t i = 0; i < n_rows; ++i) {
            const double angle = -kTwoPi * double(a) * double(i) / double(n_rows);
            const double c = std::cos(angle), s = std::sin(angle);
            for (uint32_t j = 0; j < n_cols; ++j) {
                for (uint32_t t = 0; t < batch; ++t) {
                    const double x = field[(size_t(i) * n_cols + j) * batch + t];
                    const size_t to = (size_t(a) * n_cols + j) * batch + t;
                    mid_re[to] += c * x;
                    mid_im[to] += s * x;
                }
            }
        }
    }

    // Pass two: the complex column transform, keeping col_modes bins.
    re.assign(size_t(row_modes) * col_modes * batch, 0.0);
    im.assign(re.size(), 0.0);
    for (uint32_t a = 0; a < row_modes; ++a) {
        for (uint32_t d = 0; d < col_modes; ++d) {
            for (uint32_t j = 0; j < n_cols; ++j) {
                const double angle = -kTwoPi * double(d) * double(j) / double(n_cols);
                const double c = std::cos(angle), s = std::sin(angle);
                for (uint32_t t = 0; t < batch; ++t) {
                    const size_t from = (size_t(a) * n_cols + j) * batch + t;
                    const size_t to = (size_t(a) * col_modes + d) * batch + t;
                    re[to] += c * mid_re[from] - s * mid_im[from];
                    im[to] += c * mid_im[from] + s * mid_re[from];
                }
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

}  // namespace fft
