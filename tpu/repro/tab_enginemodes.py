"""tab:enginemodes -- the three real-input row transforms against the band they produce.

Table~\\ref{tab:engines-1d} compares the three engines on the FULL spectrum, which is the one
regime an FNO never runs in: the layer keeps a band, and the band is what makes it cheap.  This
sweeps the band instead, at the two grid sizes the datasets use, so the rule the tuner applies --
"banded: recombination costs more than the smaller matmul saves" -- is measured across the band
rather than extrapolated from a single point.

Every row is checked against a float64 NumPy transform before it is timed, so a wrong answer
cannot be reported as a fast one.

Time is DEVICE time read back from this run's own traces.  At these shapes the kernel is a couple
of microseconds against a dispatch floor two orders of magnitude larger, so a wall clock measures
the floor and nothing else.
"""
import os

import jax.numpy as jnp
import numpy as np

from ..fft.core import F
from ..fft.rfft1d import rfft_low_modes
from ..experimental.fft.tricks import rfft_coupled_low_modes, rfft_reduced_low_modes
from ..experimental.bench.parse_ops import bucket_times
from .common import Bench, Paths, Timing


class EngineModes(Bench):
    TABLE = "tab:enginemodes"
    FIELDS = (["case", "grid", "n_modes", "engine", "k_cols"] + Timing.FIELDS
              + ["device_us", "mxu_us", "vpu_us", "copy_us", "dma_us",
                 "rel_err", "status", "trace"])

    K = int(os.environ.get("ENGMODES_K", "256"))
    CASES = [("Darcy-32", 32), ("NS-128", 128)]
    ENGINES = [("reim", rfft_low_modes),
               ("coupled", rfft_coupled_low_modes),
               ("reduced", rfft_reduced_low_modes)]
    # the hardware buckets parse_ops rolls XLA's hlo_category up into
    BUCKETS = ["MXU matmul", "VPU / fusion", "reshape / copy", "DMA / HBM"]

    def measure(self):
        rows = []
        for label, grid in self.CASES:
            modes = list(range(4, grid + 1, 8))
            x = jnp.asarray(np.random.default_rng(0).normal(size=(grid, self.K)).astype(np.float32))
            exact = np.fft.fft(np.asarray(x, np.float64), axis=0)
            print(f"\n  {label}: grid {grid}, K={self.K}, bands {modes}", flush=True)
            for m in modes:
                for name, fn in self.ENGINES:
                    tag = f"engmodes_{label}_{name}_{m}"
                    row = {"case": label, "grid": grid, "n_modes": m, "engine": name,
                           "k_cols": self.K, "trace": tag}
                    try:
                        re_, im_ = fn(x, m)
                        got = (np.asarray(re_, np.float64) + 1j * np.asarray(im_, np.float64))
                        row["rel_err"] = float(np.abs(got - exact[:m]).max()
                                               / np.abs(exact[:m]).max())
                        timing = self.time(lambda z, _f=fn, _m=m: _f(z, _m), (x,), tag)
                        row.update(timing.as_row())
                        row["status"] = "ok"
                    except Exception as exc:
                        row["status"] = type(exc).__name__
                        row.update({k: float("nan") for k in Timing.FIELDS})
                        row.update({"rel_err": float("nan")})
                        print(f"    m={m:<4}{name:<9}{row['status']}  {str(exc)[:40]}", flush=True)
                    rows.append(row)

        # The traces are the measurement; the wall clock above only says the call happened.
        print("\n  reading device time back from the traces", flush=True)
        for row in rows:
            for key in ("device_us", "mxu_us", "vpu_us", "copy_us", "dma_us"):
                row[key] = float("nan")
            if row["status"] != "ok":
                continue
            try:
                b = bucket_times(os.path.join(Paths.traces, row["trace"]), self.REPS)
                row["device_us"] = b.get("total", float("nan"))
                row["mxu_us"] = b.get("MXU matmul", 0.0)
                row["vpu_us"] = b.get("VPU / fusion", 0.0)
                row["copy_us"] = b.get("reshape / copy", 0.0)
                row["dma_us"] = b.get("DMA / HBM", 0.0)
            except Exception as exc:
                print(f"    {row['trace']}: {type(exc).__name__}", flush=True)

        for label, _ in self.CASES:
            sub = [r for r in rows if r["case"] == label and r["status"] == "ok"]
            if not sub:
                continue
            print(f"\n  {label}  device us per call")
            print(f"    {'m':>5}" + "".join(f"{n:>13}" for n, _ in self.ENGINES) + "   winner")
            for m in sorted({r['n_modes'] for r in sub}):
                at = {r["engine"]: r["device_us"] for r in sub if r["n_modes"] == m}
                best = min(at, key=at.get)
                print(f"    {m:>5}" + "".join(f"{at[n]:>13.2f}" for n, _ in self.ENGINES)
                      + f"   {best}")
            # where the non-matmul time goes, at the widest band of this grid
            wide = max(r["n_modes"] for r in sub)
            print(f"\n  {label}  bucket split at m={wide} (us per call)")
            print(f"    {'engine':<13}{'device':>9}{'MXU':>8}{'VPU':>8}{'copy':>8}{'DMA':>8}")
            for name, _ in self.ENGINES:
                r = next((x for x in sub if x["n_modes"] == wide and x["engine"] == name), None)
                if r:
                    print(f"    {name:<13}{r['device_us']:>9.2f}{r['mxu_us']:>8.2f}"
                          f"{r['vpu_us']:>8.2f}{r['copy_us']:>8.2f}{r['dma_us']:>8.2f}")
        return rows


TABLE = EngineModes.TABLE


def run():
    return EngineModes().run()


if __name__ == "__main__":
    run()
