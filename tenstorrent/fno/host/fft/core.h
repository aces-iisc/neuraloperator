// What every transform variant shares: the row sets, the restricted DFT matrices, the complex
// product, and the float64 references the results are judged against.
//
// This mirrors `torch_fno/fft/core.py` so the two ports can be read side by side.  The one
// structural difference is that there is no einsum here: the device performs exactly one
// operation, `Y = T @ X` on the matrix unit, so anything that contracts a middle axis has to
// become a reshape plus that one operation.  Where that happens it is spelled out.
#pragma once

#include "../device.h"
#include "../tile_matmul.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fft {

// A spectrum, as a real and an imaginary half plus what it cost to produce.
//
// The halves are flat: 1-D leaves them `n_modes x k_cols`, 2-D leaves them
// `row_modes x col_modes x batch`.  The counters are sums over every matmul the variant issued,
// which is one for the 1-D engines and four for 2-D.
struct Spectrum {
    std::vector<float> re, im;
    uint32_t mvmuls = 0;
    uint32_t unpacks = 0;
    uint32_t packs = 0;
    uint64_t clocks = 0;
    bool timed_out = false;
    std::string error;             // non-empty if the shape is one this variant cannot take
};

// Adds one matmul's instruction counts to a spectrum being built from several of them.  Returns
// false once something has gone wrong, so a caller can stop issuing.
bool accumulate(Spectrum &out, const MatmulResult &got);

Spectrum fail(const char *message);

// -- row sets --------------------------------------------------------------------------------
// Which output frequencies to evaluate.  Nothing outside the set is ever computed: the matrix is
// (len(rows) x length), not (length x length), so this is a restricted Vandermonde rather than a
// truncated FFT, and it is exact.

std::vector<uint32_t> band_rows(uint32_t n_modes);

// The band's own rows followed by each bin's mirror.  Recovering two packed spectra needs both
// Z[k] and Z[(n - k) mod n], so a band of m modes forces 2m-1 rows rather than m.
std::vector<uint32_t> mirrored_band_rows(uint32_t n, uint32_t n_modes);

// -- the matrices ----------------------------------------------------------------------------

// cos and sin of the DFT restricted to `rows`, as two separate (len(rows) x length) matrices.
// Needed separately whenever they multiply *different* operands, which is the complex case.
void rows_matrix(const std::vector<uint32_t> &rows, uint32_t length, std::vector<float> &cos_out,
                 std::vector<float> &sin_out);

// The same two stacked along the output axis, so one matmul produces both halves.  Worth it
// whenever cos and sin hit the same operand: the second matmul would re-read it.
std::vector<float> stacked_rows(const std::vector<uint32_t> &rows, uint32_t length);

// -- the complex product ---------------------------------------------------------------------

struct ComplexProduct {
    std::vector<float> re, im;
};

// (cos + i sin) @ (x_re + i x_im) in THREE real matmuls instead of four (Gauss/Karatsuba).
//
// `matmul(matrix, operand)` is the primitive to use, passed in so this stays independent of the
// shape the caller has arranged.  The three products read three different operands, so unlike
// the real case nothing can be stacked: cos and sin cannot share a pass here.
//
// The additions are host-side fp32, as the 1-D recombinations already are.  Note that this is
// not bit-identical to `torch_fno`, which forms `cos + sin` after rounding each to the operand
// format; here the sum is formed in fp32 and the unpacker rounds once, on the way into SrcB.
ComplexProduct complex_matmul(
    const std::function<std::vector<float>(const std::vector<float> &, const std::vector<float> &)>
        &matmul,
    const std::vector<float> &cos_rows, const std::vector<float> &sin_rows,
    const std::vector<float> &cos_plus_sin_rows, const std::vector<float> &x_re,
    const std::vector<float> &x_im);

// -- the references --------------------------------------------------------------------------

// The exact 1-D transform in float64, for the first `n_modes` bins of an (n x k_cols) signal.
void exact_spectrum(const std::vector<float> &signal, uint32_t n, uint32_t k_cols,
                    uint32_t n_modes, std::vector<double> &re, std::vector<double> &im);

// The exact 2-D transform in float64, for the lowest `row_modes x col_modes` bins of an
// (n_rows x n_cols x batch) real field.  Output is row_modes x col_modes x batch, matching
// `rfft2_low_modes`.
void exact_spectrum2(const std::vector<float> &field, uint32_t n_rows, uint32_t n_cols,
                     uint32_t batch, uint32_t row_modes, uint32_t col_modes,
                     std::vector<double> &re, std::vector<double> &im);

// Largest absolute error over the largest exact magnitude, across both halves together.
double spectrum_error(const Spectrum &got, const std::vector<double> &re,
                      const std::vector<double> &im);

}  // namespace fft
