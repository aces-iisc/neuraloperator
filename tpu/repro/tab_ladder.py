"""tab:ladder -- every optimisation in Section 5, each measured against the one before it.

The point of the chain is attribution.  Quoting every step against the baseline makes a step
that bought nothing look like it bought whatever the steps before it did, so each row here
reports `gain_prev`, what that step itself was worth, alongside the running `gain_base`.

The ladder is cumulative: each stage keeps everything above it and adds one thing.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import BF, F
from ..fno import layers, model
from ..experimental.fno import reference
from ..experimental.fno.gemm_variants import spectral_conv_variant
from . import common

TABLE = "tab:ladder"
FIELDS = ["case", "stage", "dtype", "batch", "params", "us_per_sample", "h1",
          "gain_prev", "gain_base", "status"]

# (stage label, what it adds).  Order matters: the chain reads top to bottom.
STAGES = ["baseline jnp.fft", "+partial DFT", "+stacking", "+blocking",
          "+ADIO layout", "+hoisted block", "+bf16", "+B*"]


def _conv(mix, stacked):
    """A spectral conv with one knob set, so a stage can be isolated."""
    return lambda w, x, m, dt=F: spectral_conv_variant(w, x, m, mix=mix, stacked=stacked, dtype=dt)


def run():
    print(f"\n### {TABLE}: incremental ladder, each step against the previous", flush=True)
    rows = []
    for label, _, channels, n_modes, batch in common.CASES:
        loaded = common.load_case(label)
        if loaded is None:
            continue
        _, _, x_test, y_test = loaded
        if not common.have_weights(label):
            print(f"  {label}: no trained weights, skipping", flush=True)
            continue
        P = common.load_weights(label)
        x = jnp.asarray(np.asarray(x_test[:batch], np.float32))
        y = np.asarray(y_test[:batch], np.float32)
        print(f"  {label}: C={channels} n_modes={n_modes} batch={batch}", flush=True)

        # each entry: (stage, params-tree, spectral conv, dtype, batch)
        plan = [
            ("baseline jnp.fft", P, None, F, batch),
            ("+partial DFT", P, _conv("naive", False), F, batch),
            ("+stacking", P, _conv("naive", True), F, batch),
            ("+blocking", P, _conv("block", True), F, batch),
            ("+ADIO layout", P, None, F, batch),            # shipped conv already stores ADIO
            ("+hoisted block", layers.prebuild_blocks(P, F), None, F, batch),
            ("+bf16", layers.prebuild_blocks(P, BF), None, BF, batch),
        ]
        case_rows = []
        for stage, params, conv, dt, bsz in plan:
            try:
                if stage == "baseline jnp.fft":
                    fn = jax.jit(lambda z, _p=params: reference.forward(_p, z, n_modes))
                elif conv is None:
                    fn = jax.jit(lambda z, _p=params, _d=dt: model.forward(_p, z, n_modes, _d))
                else:
                    fn = jax.jit(lambda z, _p=params, _c=conv, _d=dt:
                                 model.forward(_p, z, n_modes, _d, conv=_c))
                us = common.timed(fn, (x,), f"ladder_{label}_{stage}".replace(" ", "_")) / bsz
                err = common.h1(fn(x), y)
                case_rows.append({"case": label, "stage": stage, "dtype": str(dt.__name__),
                                  "batch": bsz, "params": layers.n_params(params),
                                  "us_per_sample": us, "h1": err, "status": "ok"})
                print(f"    {stage:<18}{us:>10.2f} us   H1 {err:.4e}", flush=True)
            except Exception as exc:
                why = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                case_rows.append({"case": label, "stage": stage, "dtype": str(dt.__name__),
                                  "batch": bsz, "params": 0, "us_per_sample": float("nan"),
                                  "h1": float("nan"), "status": why})
                print(f"    {stage:<18}{why:>10}  {str(exc)[:60]}", flush=True)
        common.chain(case_rows)
        rows.extend(case_rows)
    common.write(rows, "tab_ladder", FIELDS)
    return rows


if __name__ == "__main__":
    run()
