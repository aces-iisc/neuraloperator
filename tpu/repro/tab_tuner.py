"""tab:tuner -- the batch the rules predict, against the batch that measures fastest.

The batch is the one decision the planner makes about the machine rather than the data, so it is
the one worth checking against a measurement.  Two forces set it and both are properties of the
hardware: the free axis of the step-3 matmul is the batch alone -- the HLO carries the mode
indices as batch dimensions -- so the batch has to reach the edge of the array before the unit is
busy, and the working set has to stay on chip or the activations stream from memory instead.

What is swept is the configuration the batch is being chosen FOR: the shipped presets, trained
weights, the hoisted blocked filter, on the real test split.  A sweep of some other configuration
answers a question nobody asked -- a cheaper one in particular, where per-call setup dominates,
reports that the largest batch always wins no matter what the hardware does.

Time is DEVICE time, read back from this run's own traces and split by hardware unit, because
which bucket stops improving is what says why a batch is the best one.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..arch import LayerCost, ModelSpec, detect
from ..fft.core import BF, F
from ..fno import layers, model
from ..fno.autotune import AutoTune
from ..experimental.bench.parse_ops import bucket_times
from .common import CASES, Bench, Paths, Timing


class Tuner(Bench):
    TABLE = "tab:tuner"
    FIELDS = (["target", "case", "channels", "n_modes", "dtype", "batch", "b_predicted",
               "capacity", "role"] + Timing.FIELDS
              + ["device_us", "us_per_sample", "mxu_us", "vpu_us", "copy_us", "dma_us",
                 "working_mib", "status", "trace"])

    DTYPES = [(F, "float32"), (BF, "bfloat16")]

    # A dense grid, for characterising the hardware rather than tuning a model.  The wide list
    # steps in 64s so that the multiples of 64 which are NOT multiples of 128 -- 192, 320, 448
    # -- are measured too: if those cost the same as their neighbours the grain is 64, and if
    # they sit above them it is 128, which is the constant `batch_grain` asserts.  The narrow
    # list steps in 2s through the only regime where memory binds before the array fills.
    DENSE_WIDE = [64, 128, 192, 256, 320, 384, 448, 512]
    # Starts at one sample.  Navier-Stokes in bfloat16 measures fastest at the smallest
    # batch tried, so a grid that starts higher reports its own floor as the optimum.
    DENSE_NARROW = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32]

    def _grid(self, case, tuner, dtype, split):
        """The batches to measure: the planner's own few, or a dense sweep when asked."""
        if os.environ.get("TUNER_GRID") != "dense":
            return [p.batch for p in tuner.candidates(dtype)], "planner"
        dense = self.DENSE_NARROW if case.label == "NS-128" else self.DENSE_WIDE
        return [b for b in dense if b <= split], "dense"

    def measure(self):
        target = detect(os.environ.get("REPRO_TARGET"))
        print(f"  planning for {target}", flush=True)
        rows = []
        for case in CASES:
            loaded, params = case.load(), case.weights()
            if loaded is None or params is None:
                print(f"  {case.label}: no data or weights, skipping", flush=True)
                continue
            _, _, x_test, _ = loaded
            spec = ModelSpec(grid=max(x_test.shape[-2:]), channels=case.channels,
                             n_modes=case.n_modes)
            # nothing can be evaluated on more samples than the split holds
            tuner = AutoTune(target, spec, weights_are_constant=False,
                             max_batch=int(x_test.shape[0]))
            for dt, name in self.DTYPES:
                tree = layers.prebuild_blocks(params, dt)
                batches, origin = self._grid(case, tuner, name, int(x_test.shape[0]))
                predicted, why = tuner.batch(name)
                why = f"{why}; {origin} grid"
                print(f"\n  {case.label} {name}: capacity {tuner.capacity(name)}, "
                      f"predicted {predicted} ({why}), sweeping {batches}", flush=True)
                for batch in batches:
                    tag = f"tuner_{case.label}_{name}_B{batch}"
                    row = {"target": target.name, "case": case.label,
                           "channels": case.channels, "n_modes": case.n_modes,
                           "dtype": name, "batch": batch, "b_predicted": predicted,
                           "capacity": tuner.capacity(name), "role": "sweep", "trace": tag,
                           "working_mib": LayerCost(spec, target, name
                                                    ).working_bytes(batch) / 2 ** 20}
                    try:
                        x = jnp.asarray(np.asarray(x_test[:batch], np.float32))
                        fn = jax.jit(lambda z, _p=tree, _m=case.n_modes, _d=dt:
                                     model.forward(_p, z, _m, _d))
                        row.update(self.time(fn, (x,), tag, per=batch).as_row())
                        row["status"] = "ok"
                    except Exception as exc:
                        row["status"] = type(exc).__name__
                        row.update({k: float("nan") for k in Timing.FIELDS})
                        print(f"    B={batch:<6}{row['status']}  {str(exc)[:60]}", flush=True)
                    rows.append(row)

        print("\n  reading device time back from the traces", flush=True)
        for row in rows:
            for key in ("device_us", "us_per_sample", "mxu_us", "vpu_us", "copy_us", "dma_us"):
                row[key] = float("nan")
            if row["status"] != "ok":
                continue
            try:
                b = bucket_times(os.path.join(Paths.traces, row["trace"]), self.REPS)
                row["device_us"] = b.get("total", float("nan"))
                row["us_per_sample"] = row["device_us"] / row["batch"]
                row["mxu_us"] = b.get("MXU matmul", 0.0)
                row["vpu_us"] = b.get("VPU / fusion", 0.0)
                row["copy_us"] = b.get("reshape / copy", 0.0)
                row["dma_us"] = b.get("DMA / HBM", 0.0)
            except Exception as exc:
                print(f"    {row['trace']}: {type(exc).__name__}", flush=True)

        return rows + self._report(rows)

    def _report(self, rows):
        """Print each sweep and mark what won, per case and dtype."""
        selected = []
        for case in CASES:
            for _, name in self.DTYPES:
                sub = [r for r in rows
                       if r["case"] == case.label and r["dtype"] == name
                       and r["role"] == "sweep" and r["status"] == "ok"
                       and r["us_per_sample"] == r["us_per_sample"]]
                if not sub:
                    continue
                best = min(sub, key=lambda r: r["us_per_sample"])
                print(f"\n  {case.label} {name}   predicted B={best['b_predicted']}")
                print(f"    {'B':>6} {'us/sample':>11} {'vs best':>8} {'MXU':>9} {'VPU':>9} "
                      f"{'copy':>9} {'DMA':>9} {'working':>9}")
                for r in sorted(sub, key=lambda r: r["batch"]):
                    print(f"    {r['batch']:>6} {r['us_per_sample']:>11.3f} "
                          f"{r['us_per_sample'] / best['us_per_sample']:>7.2f}x "
                          f"{r['mxu_us']:>9.1f} {r['vpu_us']:>9.1f} {r['copy_us']:>9.1f} "
                          f"{r['dma_us']:>9.1f} {r['working_mib']:>8.1f}M"
                          + ("   <- best" if r is best else ""))
                print(f"    best B={best['batch']} vs predicted {best['b_predicted']} "
                      f"({best['batch'] / best['b_predicted']:.2f}x)")
                selected.append(dict(best, role="selected"))
        return selected


def run():
    return Tuner().run()


if __name__ == "__main__":
    run()
