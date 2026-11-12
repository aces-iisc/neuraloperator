#include "rfft2d.h"

namespace fft {

Spectrum rfft2_low_modes(Device &device, const std::vector<float> &field, uint32_t n_rows,
                         uint32_t n_cols, uint32_t batch, uint32_t row_modes, uint32_t col_modes,
                         uint32_t phases, uint32_t format) {
    if (row_modes == 0 || col_modes == 0) {
        return fail("a band of no modes has no transform");
    }
    if (row_modes > n_rows || col_modes > n_cols) {
        return fail("a band wider than its axis does not exist");
    }
    if (field.size() != size_t(n_rows) * n_cols * batch) {
        return fail("field size does not match its shape");
    }

    Spectrum out;

    // -- rows: real, so cos and sin stack into one pass --------------------------------------
    // An (n_rows x n_cols x batch) field read as (n_rows x n_cols*batch) is the same bytes, so
    // the operand is the input itself.
    const std::vector<float> row_matrix = stacked_rows(band_rows(row_modes), n_rows);
    const uint32_t row_wide = n_cols * batch;
    const MatmulResult rows =
        tile_matmul(device, row_matrix, field, 2 * row_modes, n_rows, row_wide, phases, format);
    if (!accumulate(out, rows)) {
        return out;
    }

    // -- regroup for the column pass ---------------------------------------------------------
    // The row pass left (2*row_modes) x (n_cols*batch).  The column pass contracts n_cols, so
    // n_cols has to lead: build (n_cols) x (row_modes*batch), which is the axis swap described
    // in the header.  This is the only data movement the split costs, and it is a host copy that
    // pack_tiles would have made anyway.
    const uint32_t col_wide = row_modes * batch;
    std::vector<float> x_re(size_t(n_cols) * col_wide);
    std::vector<float> x_im(x_re.size());
    for (uint32_t a = 0; a < row_modes; ++a) {
        for (uint32_t j = 0; j < n_cols; ++j) {
            for (uint32_t t = 0; t < batch; ++t) {
                const size_t to = size_t(j) * col_wide + size_t(a) * batch + t;
                const size_t from = size_t(j) * batch + t;
                x_re[to] = rows.y[size_t(a) * row_wide + from];
                x_im[to] = rows.y[size_t(row_modes + a) * row_wide + from];
            }
        }
    }

    // -- columns: complex, so three matmuls ---------------------------------------------------
    std::vector<float> col_cos, col_sin;
    rows_matrix(band_rows(col_modes), n_cols, col_cos, col_sin);
    std::vector<float> col_cos_plus_sin(col_cos.size());
    for (size_t i = 0; i < col_cos.size(); ++i) {
        col_cos_plus_sin[i] = col_cos[i] + col_sin[i];
    }

    bool ok = true;
    const auto matmul = [&](const std::vector<float> &matrix,
                            const std::vector<float> &operand) -> std::vector<float> {
        if (!ok) {
            return {};
        }
        const MatmulResult got =
            tile_matmul(device, matrix, operand, col_modes, n_cols, col_wide, phases, format);
        ok = accumulate(out, got);
        return ok ? got.y : std::vector<float>{};
    };

    const ComplexProduct product =
        complex_matmul(matmul, col_cos, col_sin, col_cos_plus_sin, x_re, x_im);
    if (!ok || product.re.empty()) {
        return out;
    }

    // -- back to (row_modes x col_modes x batch) ----------------------------------------------
    // Undoing the regrouping, so the result is laid out exactly as the torch port leaves it.
    out.re.assign(size_t(row_modes) * col_modes * batch, 0.0f);
    out.im.assign(out.re.size(), 0.0f);
    for (uint32_t a = 0; a < row_modes; ++a) {
        for (uint32_t d = 0; d < col_modes; ++d) {
            for (uint32_t t = 0; t < batch; ++t) {
                const size_t from = size_t(d) * col_wide + size_t(a) * batch + t;
                const size_t to = (size_t(a) * col_modes + d) * batch + t;
                out.re[to] = product.re[from];
                out.im[to] = product.im[from];
            }
        }
    }
    return out;
}

uint64_t rfft2_mvmuls(uint32_t n_rows, uint32_t n_cols, uint32_t batch, uint32_t row_modes,
                      uint32_t col_modes, uint32_t phases) {
    return expected_mvmuls(2 * row_modes, n_rows, n_cols * batch, phases) +
           3 * expected_mvmuls(col_modes, n_cols, row_modes * batch, phases);
}

}  // namespace fft
