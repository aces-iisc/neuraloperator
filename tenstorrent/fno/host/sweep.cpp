// The three engines over a ladder of grid, batch, band and fidelity phase, on the simulator.
//
// This is the Blackhole counterpart of the TPU and GPU engine sweeps, and it answers a different
// question than they do, because the simulator is explicitly not cycle-accurate and the Tensix
// model carries no instruction or cycle counter.  There is no time to report here.  What the
// simulator can answer exactly, and what silicon cannot answer better, is:
//
//   how accurate is each engine, and how many multiplies does its shape cost
//
// So the columns are `mvmuls` where the earlier sweeps had microseconds, and `rel_err` carries
// the weight.  `clocks` is recorded because it is deterministic and cheap, but it is a
// structural proxy for progress only and is never a time.
//
// Every axis is a power of two, as on the GPU, so the columns line up across grids and a missing
// cell is a shape that does not exist or does not fit rather than a gap in the sampling.
// Fidelity phases are the one axis neither earlier device had.

#include "device.h"
#include "fft/transform.h"
#include "roofline.h"
#include "numerics.h"
#include "../include/tensix.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fft;

namespace {

const uint8_t kFirmware[] = {
#include "firmware.inc"
};

// A ladder of powers of two, read from the environment when the caller wants a different reach.
std::vector<uint32_t> ladder(const char *name, std::vector<uint32_t> fallback) {
    const char *text = std::getenv(name);
    if (!text || !*text) {
        return fallback;
    }
    std::vector<uint32_t> out;
    for (const char *at = text; *at;) {
        out.push_back(uint32_t(std::strtoul(at, nullptr, 10)));
        const char *comma = std::strchr(at, ',');
        at = comma ? comma + 1 : at + std::strlen(at);
    }
    return out;
}

const char *format_name(uint32_t format) {
    return format == FMT_BF16 ? "bf16" : "tf32";
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path to libttsim.so> [out.csv]\n", argv[0]);
        return 2;
    }
    const std::string out_path = argc > 2 ? argv[2] : "results/engines_blackhole.csv";

    const std::vector<uint32_t> grids = ladder("FNO_GRIDS", {16, 32, 64, 128});
    const std::vector<uint32_t> batches = ladder("FNO_BATCHES", {16, 32, 64});
    const std::vector<uint32_t> bands = ladder("FNO_BANDS", {2, 4, 8, 16, 32, 64, 128});
    const std::vector<uint32_t> phase_list = ladder("FNO_PHASES", {1, 2, 4});
    const uint32_t format = std::getenv("FNO_TF32") ? FMT_TF32 : FMT_BF16;
    const Engine engines[] = {Engine::Reim, Engine::Coupled, Engine::Reduced};

    try {
        Device device(argv[1]);
        std::printf("vendor 0x%04x  device 0x%04x  operands %s\n", device.vendor_id(),
                    device.device_id(), format_name(format));
        device.start_firmware(std::vector<uint8_t>(kFirmware, kFirmware + sizeof(kFirmware)));
        device.clock(200000);

        FILE *csv = std::fopen(out_path.c_str(), "w");
        if (!csv) {
            throw std::runtime_error("cannot write " + out_path);
        }
        // us_1tile and us_card are projections, not measurements: the instruction count is
        // exact and the issue rate is published, but nothing here was timed.  See roofline.h.
        std::fprintf(csv, "dims,family,grid,B,n_modes,engine,phases,format,mvmuls,predicted,"
                          "unpacks,packs,us_1tile,us_card,rel_err,clocks,status\n");

        std::mt19937 gen(0);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        uint64_t rows = 0, ok = 0, absent = 0;
        double worst = 0.0;
        std::string worst_at;

        for (uint32_t grid : grids) {
            for (uint32_t batch : batches) {
                std::vector<float> signal(size_t(grid) * batch);
                for (auto &v : signal) {
                    v = dist(gen);
                }
                // The float64 reference depends only on the shape and the input, so it is
                // computed once per cell and shared by the three engines and every phase count.
                std::vector<double> ref_re, ref_im;
                uint32_t reference_band = 0;

                for (uint32_t band : bands) {
                    if (band > grid) {
                        continue;                       // a band wider than its grid does not exist
                    }
                    if (band != reference_band) {
                        exact_spectrum(signal, grid, batch, band, ref_re, ref_im);
                        reference_band = band;
                    }
                    for (Engine engine : engines) {
                        for (uint32_t phases : phase_list) {
                            const Spectrum got = transform(device, engine, signal, grid, batch,
                                                           band, phases, format);
                            ++rows;
                            const char *name = engine_name(engine);
                            const uint64_t predicted =
                                engine_mvmuls(engine, grid, batch, band, phases);

                            if (!got.error.empty()) {
                                std::fprintf(csv, "1,band,%u,%u,%u,%s,%u,%s,,%llu,,,,,,,%s\n",
                                             grid, batch, band, name, phases,
                                             format_name(format),
                                             (unsigned long long)predicted, got.error.c_str());
                                ++absent;
                                continue;
                            }
                            if (got.timed_out) {
                                std::fprintf(csv,
                                             "1,band,%u,%u,%u,%s,%u,%s,,%llu,,,,,,,timeout\n",
                                             grid, batch, band, name, phases,
                                             format_name(format),
                                             (unsigned long long)predicted);
                                continue;
                            }
                            const double error = spectrum_error(got, ref_re, ref_im);
                            const bool counted = got.mvmuls == predicted;
                            std::fprintf(csv,
                                         "1,band,%u,%u,%u,%s,%u,%s,%u,%llu,%u,%u,"
                                         "%.6g,%.6g,%.3e,%llu,%s\n",
                                         grid, batch, band, name, phases, format_name(format),
                                         got.mvmuls, (unsigned long long)predicted,
                                         got.unpacks, got.packs,
                                         roofline::micros(got.mvmuls),
                                         roofline::micros(got.mvmuls,
                                                          roofline::kTensixPerCard),
                                         error, (unsigned long long)got.clocks,
                                         counted ? "ok" : "mvmul count disagrees");
                            if (counted) {
                                ++ok;
                            }
                            if (phases == 4 && error > worst) {
                                worst = error;
                                worst_at = std::string(name) + " N=" + std::to_string(grid) +
                                           " B=" + std::to_string(batch) +
                                           " m=" + std::to_string(band);
                            }
                        }
                    }
                    std::fflush(csv);
                    std::printf("  N=%-5u B=%-5u m=%-5u done\n", grid, batch, band);
                }
            }
        }
        std::fclose(csv);
        device.stop_firmware();

        std::printf("\nwrote %s: %llu rows, %llu ok, %llu absent\n", out_path.c_str(),
                    (unsigned long long)rows, (unsigned long long)ok,
                    (unsigned long long)absent);
        std::printf("worst rel_err at four phases: %.3e  (%s)\n", worst, worst_at.c_str());
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
