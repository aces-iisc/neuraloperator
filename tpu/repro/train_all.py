"""Train one model per dataset per engine, once, and keep the weights.

Every table that needs a trained network reads these files, so a full reproduction trains the
set once rather than once per table.

Each dataset is trained on the objective its config in config/ specifies -- h1 for Darcy and
Navier-Stokes, l2 for Burgers -- and since the evaluation metric follows the loss, the saved
checkpoint is the best one by that same measure.
"""
import os
import time

from ..fno import layers, model, train
from ..fno.spectral import separable_conv
from ..experimental.fno import reference
from .common import CASES, Paths


class Trainer:
    """Produces the weights the rest of the reproduction consumes."""

    EPOCHS = int(os.environ.get("REPRO_EPOCHS", "60"))
    BATCH = int(os.environ.get("REPRO_TRAIN_BATCH", "32"))

    # "ours" is the shipped spectral conv; "jnp" is the complex64 reference the results
    # table compares against.  Both are trained here so no table trains its own.
    ENGINES = [("ours", model.forward),
               ("jnp", reference.forward),
               # F-FNO is a different parameterisation, so it is trained separately rather
               # than evaluated on the dense weights
               ("ffno", lambda P, x, m: model.forward(P, x, m, conv=separable_conv))]

    def run(self):
        print(f"\n### trained weights ({self.EPOCHS} epochs, objective per config)", flush=True)
        Paths.ensure()
        produced = []
        for case in CASES:
            loaded = None
            for engine, forward in self.ENGINES:
                path = Paths.weight_file(case.label, engine)
                if os.path.exists(path):
                    print(f"  {case.label}/{engine}: already trained, reusing", flush=True)
                    produced.append(path)
                    continue
                loaded = loaded or case.load()
                if loaded is None:
                    break
                produced.append(self._train_one(case, engine, forward, loaded, path))
        return [p for p in produced if p]

    def _train_one(self, case, engine, forward, loaded, path):
        x_tr, y_tr, x_te, y_te = loaded
        init = (layers.init_fno_separable if engine == "ffno" else None)
        start = time.perf_counter()
        try:
            params, err, _ = train.train(x_tr, y_tr, x_te, y_te, n_modes=case.n_modes,
                                         channels=case.channels, n_layers=4,
                                         steps=self.EPOCHS, batch=self.BATCH,
                                         loss=case.loss, fwd=forward, init=init,
                                         verbose=False)
        except Exception as exc:
            print(f"  {case.label}/{engine}: {type(exc).__name__} {str(exc)[:60]}", flush=True)
            return None
        model.save_params(params, path)
        print(f"  {case.label}/{engine}: {case.loss} {err:.4e}  "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
        return path


TABLE = None            # produces weights, not a table


def run():
    return Trainer().run()


if __name__ == "__main__":
    run()
