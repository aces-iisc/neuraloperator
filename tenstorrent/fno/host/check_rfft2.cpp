// The 2-D transform against a float64 reference, shape by shape.
//
//   ./check_rfft2 <libttsim.so>
//
// Two things are checked per case, and they fail for different reasons.
//
//   the instruction count, exactly.  `rfft2_mvmuls` derives it from the shape alone, so a loop
//   bound that is off by a tile, or a column pass that was grouped the wrong way, shows up here
//   even when the numbers look plausible.
//
//   the relative error against the exact partial DFT in float64.  The operands are bf16 or tf32
//   and the 2-D path rounds twice -- once into the row pass, once into the column pass -- so the
//   error is larger than the 1-D sweep's at the same format, and the tolerance says so per case
//   rather than globally.
//
// What this does not check is the band selection being free: that is `rfft2_mvmuls` growing with
// the modes rather than with the grid, which the printed counts show directly.
//
// Three numbers on each line are times, and none of them is a measurement of the device.
//
//   us_1tile / us_card   the roofline: the exact MVMUL count divided by the published issue rate,
//                        on one Tensix and on a 120-Tensix card.  A lower bound on silicon, since
//                        it assumes the matrix unit never waits.  See roofline.h.
//   sim_us               host wallclock for the simulated run.  This is the x86 host interpreting
//                        RISC-V and Tensix semantics, so it scales with instructions retired and
//                        says nothing about how fast the hardware would be.  It is reported
//                        because it is the only quantity here that was actually timed.

#include "device.h"
#include "numerics.h"
#include "fft/rfft1d.h"
#include "fft/rfft2d.h"
#include "roofline.h"
#include "../include/tensix.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fft;

namespace {

const uint8_t kFirmware[] = {
#include "firmware.inc"
};

struct Case {
    const char *name;
    uint32_t n_rows, n_cols, batch, row_modes, col_modes, phases, format;
    double tol;
};

// Shapes chosen so each axis fails on its own: one square case per grid, then the band narrowed
// on each axis in turn, then the batch, then a non-square field, then the fidelity axis.
// The tolerances are measured bounds with roughly a factor of two of headroom, not derived
// ones.  Their job is to catch a structural break -- a loop bound, a regrouping, a dropped
// Karatsuba term -- not to certify the accuracy, which is what the sweep reports.  See the
// FNO_STAGES diagnostic below for where the error actually comes from.
const Case kCases[] = {
    {"half band, 16x16", 16, 16, 1, 8, 8, 1, FMT_BF16, 1.5e-1},
    {"half band, 32x32", 32, 32, 1, 16, 16, 1, FMT_BF16, 1.5e-1},
    {"half band, 64x64", 64, 64, 1, 32, 32, 1, FMT_BF16, 1.5e-1},
    {"narrow band, 64x64", 64, 64, 1, 8, 8, 1, FMT_BF16, 1.5e-1},
    {"rows wide, cols narrow", 64, 64, 1, 32, 8, 1, FMT_BF16, 1.5e-1},
    {"rows narrow, cols wide", 64, 64, 1, 8, 32, 1, FMT_BF16, 1.5e-1},
    {"full band on rows", 32, 32, 1, 32, 16, 1, FMT_BF16, 1.5e-1},
    {"non-square field", 32, 64, 1, 16, 32, 1, FMT_BF16, 1.5e-1},
    {"batch of 8", 32, 32, 8, 16, 16, 1, FMT_BF16, 1.5e-1},
    {"batch of 8, 64x64", 64, 64, 8, 16, 16, 1, FMT_BF16, 1.5e-1},
    {"four phases", 64, 64, 1, 32, 32, 4, FMT_BF16, 3e-2},
    {"tf32, four phases", 64, 64, 1, 32, 32, 4, FMT_TF32, 1e-2},
};

const char *format_name(uint32_t format) {
    return format == FMT_BF16 ? "bf16" : "tf32";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libttsim.so>\n", argv[0]);
        return 2;
    }

    try {
        Device device(argv[1]);
        std::printf("vendor 0x%04x  device 0x%04x\n\n", device.vendor_id(), device.device_id());
        device.start_firmware(std::vector<uint8_t>(kFirmware, kFirmware + sizeof(kFirmware)));
        device.clock(200000);

        std::mt19937 gen(7);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        unsigned failures = 0;

        for (const Case &test : kCases) {
            std::vector<float> field(size_t(test.n_rows) * test.n_cols * test.batch);
            for (float &value : field) {
                value = dist(gen);
            }

            const auto started = std::chrono::steady_clock::now();
            const Spectrum got =
                rfft2_low_modes(device, field, test.n_rows, test.n_cols, test.batch,
                                test.row_modes, test.col_modes, test.phases, test.format);
            const double sim_us =
                std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() -
                                                          started)
                    .count();
            const uint64_t predicted = rfft2_mvmuls(test.n_rows, test.n_cols, test.batch,
                                                    test.row_modes, test.col_modes, test.phases);

            std::printf("  %-24s %3ux%-3u b%-2u m %2ux%-2u p%u %s  ", test.name, test.n_rows,
                        test.n_cols, test.batch, test.row_modes, test.col_modes, test.phases,
                        format_name(test.format));

            if (!got.error.empty()) {
                std::printf("refused: %s\n", got.error.c_str());
                ++failures;
                continue;
            }
            if (got.timed_out) {
                std::printf("timed out\n");
                ++failures;
                continue;
            }

            std::vector<double> ref_re, ref_im;
            exact_spectrum2(field, test.n_rows, test.n_cols, test.batch, test.row_modes,
                            test.col_modes, ref_re, ref_im);
            const double error = spectrum_error(got, ref_re, ref_im);
            const bool counts_agree = got.mvmuls == predicted;

            std::printf("mvmul %5u  unpack %5u  pack %4u  rel_err %.3e  "
                        "us_1tile %7.3f  us_card %7.4f  sim %8.1f us",
                        got.mvmuls, got.unpacks, got.packs, error,
                        roofline::micros(got.mvmuls),
                        roofline::micros(got.mvmuls, roofline::kTensixPerCard), sim_us);
            if (!counts_agree) {
                std::printf("  COUNT expected %llu", (unsigned long long)predicted);
                ++failures;
            } else if (error > test.tol) {
                std::printf("  OVER tol %.1e", test.tol);
                ++failures;
            } else {
                std::printf("  ok");
            }
            if (std::getenv("FNO_STAGES")) {
                // The row pass on its own is exactly rfft_low_modes over the field read as
                // (n_rows x n_cols*batch), so it can be run and judged against the 1-D
                // reference.  Comparing the two errors says whether the 2-D figure above comes
                // from the real row pass or from the complex column pass, where the operands are
                // rounded a second time and Karatsuba's `c - a - b` cancels.
                const Spectrum rows =
                    rfft_low_modes(device, field, test.n_rows, test.n_cols * test.batch,
                                   test.row_modes, test.phases, test.format);
                std::vector<double> row_re, row_im;
                exact_spectrum(field, test.n_rows, test.n_cols * test.batch, test.row_modes,
                               row_re, row_im);
                std::printf("      row pass alone: rel_err %.3e over %u mvmul\n",
                            spectrum_error(rows, row_re, row_im), rows.mvmuls);
            }
            std::printf("\n");
        }

        std::printf("\n%s\n", failures == 0 ? "all cases ok" : "cases failed");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
