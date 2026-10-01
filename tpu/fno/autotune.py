"""Predicts num_modes from the data's spectrum and the batch from the layer's shapes."""
import numpy as np

MXU = 128                 # matrix unit edge
VMEM = 128 * 1024 * 1024  # v5e vector memory
ACC_BYTES = 4             # accumulation stays float32 whatever the operands are
LIVE = 3                  # live accumulator buffers per element, calibrated on NS-128
BYTES = {"float32": 4, "bfloat16": 2, "int8": 1}


def predict_modes(targets, tau=0.999, n_sample=64, round_to=8):
    """Smallest mode band holding a fraction tau of the radial spectral energy.

    `targets` is a stack of output fields (S, H, W). Costs one FFT over n_sample of them,
    so it does not scale with the training set and needs no model."""
    y = np.asarray(targets[:n_sample], dtype=np.float64)
    spec = np.abs(np.fft.fft2(y, axes=(1, 2))) ** 2
    n = spec.shape[1]
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij")
    radius = np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
    total = spec.sum()
    cutoff = n // 2
    for r in range(1, n // 2 + 1):
        if spec[:, radius <= r].sum() / total >= tau:
            cutoff = r
            break
    modes = 2 * cutoff                                  # a band of m modes reaches radius m/2
    modes = int(round_to * max(1, round(modes / round_to)))
    return min(modes, n)


def energy_curve(targets, bands, n_sample=64):
    """Fraction of radial spectral energy inside each band, for reporting."""
    y = np.asarray(targets[:n_sample], dtype=np.float64)
    spec = np.abs(np.fft.fft2(y, axes=(1, 2))) ** 2
    n = spec.shape[1]
    ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij")
    radius = np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
    total = spec.sum()
    return {m: float(spec[:, radius <= m // 2].sum() / total) for m in bands}


def predict_batch(channels, n_rows, n_cols, dtype="float32"):
    """Where the batch sweep starts: the unit width for a narrow tile, else the VMEM capacity."""
    w = BYTES[dtype]
    per_elem = w + LIVE * ACC_BYTES
    capacity = max(1, VMEM // (channels * n_rows * n_cols * per_elem))
    if 2 * channels < MXU:
        # A narrow tile starves the free axis, so start at the unit width. On these grids the
        # VMEM bound is not what ends the gain -- it overshoots -- so it is capped here and the
        # sweep of neighbours_mul is left to explore above it.
        return MXU
    return int(capacity)


def neighbours_mul(value, n=2, lo=1, hi=None):
    """The value and n powers of two either side. For the batch, which spans decades."""
    out = []
    for k in range(-n, n + 1):
        v = int(round(value * (2.0 ** k)))
        v = max(lo, v if hi is None else min(v, hi))
        if v not in out:
            out.append(v)
    return sorted(out)


def neighbours_add(value, step=8, n=2, lo=None, hi=None):
    """The value and n steps either side. For the mode count, which moves in small units."""
    lo = lo if lo is not None else step
    out = []
    for k in range(-n, n + 1):
        v = value + k * step
        v = max(lo, v if hi is None else min(v, hi))
        if v not in out:
            out.append(v)
    return sorted(out)


# The transform method, settled by measurement rather than searched.  Sweeping all three over
# n_modes x N (both powers of two, n_modes <= N <= 4096, each cell at its own best batch), the
# banded reim DFT wins 53 of 60 cells.  The seven it loses are all within 1.02x and sit at the
# extremes -- either n_modes = 4, where the row pass is too small to matter and the three methods
# tie, or the full spectrum.  There is one real crossover, at n_modes = N = 1024, where the
# even/odd reduction is 1.4% faster on device time because its matrix is half the height; an FNO
# never runs there, since the band is what makes the layer cheap in the first place.
#
# The reduction is also a liability at the shapes that matter: at grid 128 its recombination
# gathers lower to a while loop, and it measures 12,156 us against reim's 349 with an identical
# 272 us of matmul.  All three agree to float32 roundoff, so the ordering is speed alone.
# See tpu/results/modesweep/ and tpu/results/rerun/.
METHODS = ("reim", "coupled", "reduced")


def predict_method(n_grid=None, n_modes=None):
    """Which transform to run.

    Constant: the only measured crossover is at n_modes = N, which is outside the regime an FNO
    operates in.  The alternatives stay in METHODS because the sweep is cheap insurance if the
    band ever grows towards the grid.
    """
    return "reim"
