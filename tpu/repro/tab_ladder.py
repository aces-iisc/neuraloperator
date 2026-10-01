"""tab:ladder -- every optimisation in Section 5, each measured against the one before it.

Three kinds of change, kept apart because they are not comparable:

  structural   exact rewrites of the same arithmetic -- these chain, and each row reports what
               that step alone was worth
  precision    f32 against bf16, the same algorithm at half the activation width.  It is a
               column rather than a rung: it applies to every structural stage independently
               and it is not exact, so folding it into the chain would hide which rows are
               output-preserving
  model        a different parameterisation (F-FNO), with its own weights and its own accuracy

The chain exists for attribution.  Quoting every step against the baseline makes a step that
bought nothing look like it bought whatever the steps before it did, so each row carries
`gain_prev`, what that step itself was worth, next to the running `gain_base`.

The ladder is cumulative: each stage keeps everything above it and adds one thing.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import BF, F
from ..fno import layers, model
from ..fno.spectral import separable_conv
from ..experimental.fno import reference
from ..experimental.fno.gemm_variants import spectral_conv_variant
from .common import CASES, Bench, Timing


class Ladder(Bench):
    TABLE = "tab:ladder"
    FIELDS = (["case", "stage", "kind", "loss", "dtype", "batch", "params", "param_ratio"]
              + Timing.FIELDS
              + ["error", "h1", "l2", "gain_prev", "gain_base", "status"])

    @staticmethod
    def _variant(mix, stacked):
        """A spectral conv with one knob set, so a single stage can be isolated."""
        return lambda w, x, m, dt=F: spectral_conv_variant(w, x, m, mix=mix, stacked=stacked,
                                                           dtype=dt)

    def _plan(self, params):
        """(stage, params, spectral conv, dtype).  None means the shipped conv."""
        return [
            ("baseline jnp.fft", params, "reference", F),
            ("+partial DFT", params, self._variant("naive", False), F),
            ("+stacking", params, self._variant("naive", True), F),
            ("+blocking", params, self._variant("block", True), F),
            ("+ADIO layout", params, None, F),
            ("+hoisted block", layers.prebuild_blocks(params, F), None, F),
        ]

    # Everything in _plan preserves the output exactly, so those stages chain.  F-FNO does not:
    # it is a different parameterisation with its own weights and its own accuracy, so it is
    # measured against the dense model at the SAME precision rather than folded into the chain.
    def _model_choices(self):
        return [("F-FNO", separable_conv, "+hoisted block")]

    def measure(self):
        rows = []
        for case in CASES:
            loaded = case.load()
            if loaded is None:
                continue
            params = case.weights()
            ffno_params = case.weights("ffno")
            if params is None:
                print(f"  {case.label}: no trained weights yet, skipping", flush=True)
                continue
            _, _, x_test, y_test = loaded
            x = jnp.asarray(np.asarray(x_test[:case.batch], np.float32))
            y = np.asarray(y_test[:case.batch], np.float32)
            print(f"  {case.label}: C={case.channels} m={case.n_modes} B={case.batch}",
                  flush=True)

            case_rows = []
            dense_params = layers.n_params(params)
            for stage, tree, conv, dtype in self._plan(params):
              for prec_name, prec in (("f32", F), ("bf16", BF)):
                tree_p = (layers.prebuild_blocks(params, prec)
                          if stage == "+hoisted block" else tree)
                row = {"case": case.label, "stage": stage, "kind": "structural",
                       "loss": case.loss, "dtype": prec_name, "batch": case.batch,
                       "params": layers.n_params(tree_p), "param_ratio": 1.0}
                try:
                    fn = self._compile(tree_p, conv, prec, case.n_modes)
                    timing = self.time(
                        fn, (x,), f"ladder_{case.label}_{stage}_{prec_name}".replace(" ", "_"),
                        per=case.batch)
                    row.update(timing.as_row())
                    both = self.errors(fn(x), y)
                    row.update(both)
                    row["error"] = both[case.loss]       # the metric it was trained for
                    row["status"] = "ok"
                    print(f"    {stage:<18}{prec_name:<6}{timing}  "
                          f"{case.loss} {row['error']:.4e}", flush=True)
                except Exception as exc:
                    row["status"] = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    row.update({"error": float("nan"), "h1": float("nan"),
                                "l2": float("nan")})
                    row.update({k: float("nan") for k in Timing.FIELDS})
                    print(f"    {stage:<18}{prec_name:<6}{row['status']:>12}  "
                          f"{str(exc)[:50]}", flush=True)
                case_rows.append(row)
            # chain within each precision: a stage is compared to the same stage one rung down,
            # at the same width, never across widths
            for prec_name in ("f32", "bf16"):
                self.chain([r for r in case_rows if r["dtype"] == prec_name])
            rows.extend(case_rows)

            # the model choice, at each precision, against the dense row it replaces
            if ffno_params is None:
                print("    F-FNO           no trained weights, skipping", flush=True)
                continue
            by_stage = {(r["stage"], r["dtype"]): r for r in case_rows}
            for stage, conv, against in self._model_choices():
              for prec_name, dtype in (("f32", F), ("bf16", BF)):
                row = {"case": case.label, "stage": stage, "kind": "model",
                       "loss": case.loss, "dtype": prec_name, "batch": case.batch,
                       "params": layers.n_params(ffno_params),
                       "param_ratio": dense_params / max(1, layers.n_params(ffno_params))}
                try:
                    fn = self._compile(ffno_params, conv, dtype, case.n_modes)
                    timing = self.time(
                        fn, (x,), f"ladder_{case.label}_{stage}_{prec_name}".replace(" ", "_"),
                        per=case.batch)
                    row.update(timing.as_row())
                    both = self.errors(fn(x), y)
                    row.update(both)
                    row["error"] = both[case.loss]
                    row["status"] = "ok"
                    peer = by_stage.get((against, prec_name), {})
                    base = peer.get("us_mean")
                    row["gain_prev"] = (base / timing.mean) if base else float("nan")
                    first = next((r["us_mean"] for r in case_rows
                                  if r["dtype"] == prec_name and r.get("us_mean")), None)
                    row["gain_base"] = (first / timing.mean) if first else float("nan")
                    print(f"    {stage:<18}{prec_name:<6}{timing}  "
                          f"{case.loss} {row['error']:.4e}  "
                          f"{row['param_ratio']:.1f}x fewer params  "
                          f"({row['gain_prev']:.3f}x vs dense)", flush=True)
                except Exception as exc:
                    row["status"] = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                    row.update({k: float("nan") for k in Timing.FIELDS})
                    row.update({"error": float("nan"), "h1": float("nan"), "l2": float("nan"),
                                "gain_prev": float("nan"), "gain_base": float("nan")})
                    print(f"    {stage:<18}{prec_name:<6}{row['status']:>12}  "
                          f"{str(exc)[:50]}", flush=True)
                rows.append(row)
        return rows

    @staticmethod
    def _compile(tree, conv, dtype, n_modes):
        if conv == "reference":
            return jax.jit(lambda z: reference.forward(tree, z, n_modes))
        if conv is None:
            return jax.jit(lambda z: model.forward(tree, z, n_modes, dtype))
        return jax.jit(lambda z: model.forward(tree, z, n_modes, dtype, conv=conv))


TABLE = Ladder.TABLE


def run():
    return Ladder().run()


if __name__ == "__main__":
    run()
