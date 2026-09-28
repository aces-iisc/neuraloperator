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
