#include "transform.h"

#include "rfft1d.h"
#include "tricks.h"

namespace fft {

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
    switch (engine) {
        case Engine::Reim:
            return rfft_low_modes(device, signal, n, k_cols, n_modes, phases, format);
        case Engine::Coupled:
            return rfft_coupled_low_modes(device, signal, n, k_cols, n_modes, phases, format);
        case Engine::Reduced:
            return rfft_reduced_low_modes(device, signal, n, k_cols, n_modes, phases, format);
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

}  // namespace fft
