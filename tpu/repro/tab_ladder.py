"""tab:ladder -- every optimisation, from the shipped default to the fastest configuration.

The chain starts where a user starts: the reference transform, on the shipped weights, at the
batch the model trained with.  Each rung then adds one thing and keeps everything above it, so
the step gains multiply out to the net figure on the last row.

Three kinds of change, kept in separate groups because they are not the same claim:

  transform    exact rewrites of the same arithmetic, all at the training batch.  Every row in
               this group reproduces the baseline output, so the whole group is free.
  execution    the same arithmetic again, issued at a different batch and a different operand
               width.  The batch is output-preserving; the width is not, and its error is shown.
  filter       a different parameterisation with its own weights, measured at the configuration
               the group above ends on.  Its error is not expected to match.

The batch is carried on every row, since it is the quantity the execution group changes and the
one that decides how much of the net figure is available.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import BF, F
from ..fno import layers, model
from ..fno.spectral import cp_conv, separable_conv
from ..experimental.fno import reference
from ..experimental.fno.gemm_variants import spectral_conv_variant
from .common import CASES, Bench, Timing


class Ladder(Bench):
    TABLE = "tab:ladder"
    FIELDS = (["case", "group", "stage", "dtype", "batch", "params", "param_ratio"]
              + Timing.FIELDS
              + ["loss", "error", "h1", "l2", "gain_prev", "gain_base", "status"])

    # what an untouched checkpoint is evaluated at, and what the chain therefore starts from
    TRAIN_BATCH = {"Burgers": 16, "Darcy-32": 8, "Darcy-64": 8, "NS-128": 8}
    # The best batch tracks the matrix unit's edge, so a part with a wider unit wants a wider
    # sweep: LADDER_SWEEP and LADDER_SWEEP_NS extend the range without touching the v5e defaults.
    SWEEP = {"NS-128": [int(v) for v in
                        os.environ.get("LADDER_SWEEP_NS", "1,2,4,8,16").split(",")]}
    DEFAULT_SWEEP = [int(v) for v in
                     os.environ.get("LADDER_SWEEP", "8,16,32,64,128,256").split(",")]

    @staticmethod
    def _variant(mix, stacked):
        return lambda w, x, m, dt=F: spectral_conv_variant(w, x, m, mix=mix, stacked=stacked,
                                                           dtype=dt)

    @staticmethod
    def _compile(tree, conv, dtype, n_modes):
        if conv == "reference":
            return jax.jit(lambda z: reference.forward(tree, z, n_modes))
        if conv is None:
            return jax.jit(lambda z: model.forward(tree, z, n_modes, dtype))
        return jax.jit(lambda z: model.forward(tree, z, n_modes, dtype, conv=conv))

    def _best_batch(self, case, tree, dtype, x_test):
        """The batch with the lowest device time per sample -- what the tuner selects."""
        best, best_us = self.TRAIN_BATCH[case.label], float("inf")
        for batch in self.SWEEP.get(case.label, self.DEFAULT_SWEEP):
            if batch > x_test.shape[0]:
                continue
            try:
                x = jnp.asarray(np.asarray(x_test[:batch], np.float32))
                us = self.time(self._compile(tree, None, dtype, case.n_modes), (x,),
                               f"ladder_sweep_{case.label}_{dtype.__name__}_{batch}",
                               per=batch, reps=5).mean
            except Exception:
                continue
            if us < best_us:
                best, best_us = batch, us
        return best

    def measure(self):
        rows = []
        for case in CASES:
            loaded = case.load()
            params = case.weights()
            if loaded is None or params is None:
                print(f"  {case.label}: no data or weights, skipping", flush=True)
                continue
            _, _, x_test, y_test = loaded
            btr = self.TRAIN_BATCH[case.label]
            hoist = {d: layers.prebuild_blocks(params, d) for d in (F, BF)}
            b32 = self._best_batch(case, hoist[F], F, x_test)
            b16 = self._best_batch(case, hoist[BF], BF, x_test)
            dense_params = layers.n_params(params)
            print(f"  {case.label}: B_train={btr}  B*(f32)={b32}  B*(bf16)={b16}", flush=True)

            plan = [
                ("transform", "baseline jnp.fft", params, "reference", F, btr),
                ("transform", "+partial DFT", params, self._variant("naive", False), F, btr),
                ("transform", "+stacking", params, self._variant("naive", True), F, btr),
                ("transform", "+blocking", params, self._variant("block", True), F, btr),
                ("transform", "+ADIO layout", params, None, F, btr),
                ("transform", "+hoisted block", hoist[F], None, F, btr),
                ("execution", "+B*", hoist[F], None, F, b32),
                ("execution", "+bfloat16", hoist[BF], None, BF, b16),
            ]
            # the filter choices are measured where the chain ends, so the comparison is fair
            for name, conv, engine in (("F-FNO", separable_conv, "ffno"),
                                       ("TFNO", cp_conv, "tfno")):
                tree = case.weights(engine)
                # prebuild_blocks assembles the DENSE blocked filter; the variants carry
                # their own structure, so they are passed through as trained
                if tree is not None:
                    plan.append(("filter", name, tree, conv, BF, b16))

            first = prev = None
            for group, stage, tree, conv, dtype, batch in plan:
                row = {"case": case.label, "group": group, "stage": stage,
                       "dtype": "f32" if dtype is F else "bf16", "batch": batch,
                       "loss": case.loss, "params": layers.n_params(tree),
                       "param_ratio": dense_params / max(1, layers.n_params(tree))}
                try:
                    fn = self._compile(tree, conv, dtype, case.n_modes)
                    x = jnp.asarray(np.asarray(x_test[:batch], np.float32))
                    tag = f"ladder_{case.label}_{stage}_{row['dtype']}".replace(" ", "_")
                    timing = self.time(fn, (x,), tag, per=batch)
                    row.update(timing.as_row())
                    both = self.errors_chunked(fn, x_test, y_test, batch)
                    row.update(both)
                    row["error"] = both[case.loss]
                    first = first or timing.mean
                    # the filter group replaces the last rung rather than extending the chain
                    row["gain_prev"] = (prev / timing.mean) if (prev and group != "filter") else 1.0
                    row["gain_base"] = first / timing.mean
                    if group != "filter":
                        prev = timing.mean
                    row["status"] = "ok"
                    print(f"    {group:<10}{stage:<18}{row['dtype']:<6}B={batch:<5}{timing}  "
                          f"{case.loss} {row['error']:.4e}  net {row['gain_base']:.2f}x",
                          flush=True)
                except Exception as exc:
                    row["status"] = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    row.update({k: float("nan") for k in Timing.FIELDS})
                    row.update({"error": float("nan"), "h1": float("nan"), "l2": float("nan"),
                                "gain_prev": float("nan"), "gain_base": float("nan")})
                    print(f"    {group:<10}{stage:<18}{row['status']:>12}  {str(exc)[:46]}",
                          flush=True)
                rows.append(row)
        return rows


TABLE = Ladder.TABLE


def run():
    return Ladder().run()


if __name__ == "__main__":
    run()
