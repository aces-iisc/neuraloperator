"""Radial spectral energy of each dataset's targets, and the band the rule picks from it.

Fills tab:modeenergy.  This is the one tuner decision taken from the data rather than the
machine, so it needs neither a TPU nor trained weights -- only NumPy and the datasets.

Pure NumPy over the training targets, so it needs neither a TPU nor the trained weights: this is
the one tuner decision that comes from the data rather than the machine.
"""
import importlib.util, os, sys
import numpy as np

_spec = importlib.util.spec_from_file_location(
    "tpu_data", os.path.join(os.path.dirname(__file__), "..", "..", "fno", "data.py"))
tdata = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tdata)

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
BANDS = (8, 16, 24, 32, 48, 64)
CASES = [("Burgers", lambda: tdata.load_burgers(ROOT)),
         ("Darcy-32", lambda: tdata.load_darcy(ROOT, 32)),
         ("Darcy-64", lambda: tdata.load_darcy(ROOT, 64)),
         ("NS-128", lambda: tdata.load_ns(ROOT, 128, n_train=1000, n_test=200))]


def curve(targets, n_sample=64):
    y = np.asarray(targets[:n_sample], np.float64)
    y = y[:, 0] if y.ndim == 4 else y
    power = np.abs(np.fft.fft2(y, axes=(1, 2))) ** 2
    rows, cols = power.shape[1], power.shape[2]
    ky, kx = np.meshgrid(np.fft.fftfreq(rows) * rows,
                         np.fft.fftfreq(cols) * cols, indexing="ij")
    radius = np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
    total = power.sum()
    frac = {m: float(power[:, radius <= m // 2].sum() / total) for m in BANDS
            if m // 2 <= max(rows, cols) // 2}
    # the rule: smallest radius holding 0.999 of the energy, band = 2r rounded to 8
    cutoff = min(rows, cols) // 2
    for r in range(1, min(rows, cols) // 2 + 1):
        if power[:, radius <= r].sum() / total >= 0.999:
            cutoff = r
            break
    m_star = min(int(8 * max(1, round(2 * cutoff / 8))), min(rows, cols))
    return frac, m_star


def main():
    print(f"{'dataset':<10}" + "".join(f"{('E(' + str(m) + ')'):>10}" for m in BANDS) + f"{'m*':>6}")
    for label, loader in CASES:
        _, ytr, _, _ = loader()
        frac, m_star = curve(ytr)
        cells = "".join(f"{frac[m]*100:>10.3f}" if m in frac else f"{'---':>10}" for m in BANDS)
        print(f"{label:<10}{cells}{m_star:>6}")


if __name__ == "__main__":
    main()
