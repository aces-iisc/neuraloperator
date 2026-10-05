"""tab:cprank -- TFNO accuracy and device time as the CP rank varies.

Rank is the one knob that moves TFNO along the line between its own cost and F-FNO's.  The CP
contraction runs over the rank axis, so its arithmetic grows linearly with the rank while the
dense filter's is fixed; the ratio against the dense filter is 2R/C, which puts the dense cost
at R = C/2 and F-FNO's at R = C.

Whether the device charges for that growth is a separate question.  The rank is the contracted
axis of the second product and is narrower than the matrix unit at the rank the model ships
with, so the padding is paid whatever the rank is and the arithmetic ratio is an upper bound on
what the sweep should show.

R = C is also the smallest rank at which a CP filter can represent an F-FNO one: F-FNO keeps a
dense channel mixing, so unfolding its filter along either channel axis gives rank C.

One model is trained per rank on the objective the dataset's config specifies, then timed with
a trace, so the accuracy and the time come from the same weights.
"""
import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import F
from ..fno import layers, model, train
from ..fno.spectral import cp_conv, separable_conv
from .common import BY_NAME, Bench, Paths, Timing


class CpRank(Bench):
    """One row per rank, plus the dense and F-FNO filters the sweep is placed between."""

    TABLE = "tab:cprank"
    FIELDS = (["case", "filter", "rank", "params", "filter_params", "param_ratio"]
              + Timing.FIELDS
              + ["error", "h1", "l2", "train_s", "vs_dense", "status"])

    CASE = os.environ.get("CPRANK_CASE", "Darcy-64")
    RANKS = [int(r) for r in os.environ.get("CPRANK_RANKS", "8,12,16,24,32").split(",")]
    EPOCHS = int(os.environ.get("REPRO_EPOCHS", "300"))
    TRAIN_BATCH = int(os.environ.get("REPRO_TRAIN_BATCH", "32"))

    @staticmethod
    def _filter_params(tree):
        """Free parameters in the spectral filters only.

        The lift, projection and skip paths are identical in every row, so a ratio taken over
        the whole model understates what the factorisation saves.

        F-FNO stores its weights already blocked as [[re, -im], [im, re]], which holds each real
        and imaginary entry twice, so its free count is half of what the array holds.  The dense
        and CP filters store one copy of each entry.
        """
        total = 0
        for block in tree["blocks"]:
            held = layers.n_params(block["filter"])
            total += held // 2 if "k_block" in block["filter"] else held
        return total

    @staticmethod
    def _init(rank):
        """The CP initialiser bound to one rank.  Its scale already tracks the rank."""
        return functools.partial(layers.init_fno_cp, rank=rank)

    @classmethod
    def _weight_path(cls, case, rank):
        """Rank 8 is what init_fno_cp defaults to, so the shipped checkpoint is that rank."""
        engine = "tfno" if rank == 8 else f"tfno_r{rank}"
        return Paths.weight_file(case.label, engine)

    def _weights(self, case, loaded, rank):
        """Load the checkpoint for this rank, training it first if it does not exist yet."""
        path = self._weight_path(case, rank)
        if os.path.exists(path):
            tree = model.load_params(path, channels=case.channels, n_modes=case.n_modes,
                                     init=self._init(rank))
            return tree, float("nan")

        x_train, y_train, x_test, y_test = loaded
        forward = lambda P, x, m: model.forward(P, x, m, conv=cp_conv)
        start = time.perf_counter()
        tree, _, _ = train.train(x_train, y_train, x_test, y_test, n_modes=case.n_modes,
                                 channels=case.channels, n_layers=4, steps=self.EPOCHS,
                                 batch=self.TRAIN_BATCH, loss=case.loss, fwd=forward,
                                 init=self._init(rank), verbose=False)
        train_s = time.perf_counter() - start
        model.save_params(tree, path)
        return tree, train_s

    def _row(self, case, name, rank, tree, train_s, conv, x, y, dense_params, dense_us):
        """Time one filter on the trained weights and read both metrics off the same pass."""
        filter_params = self._filter_params(tree)
        row = {"case": case.label, "filter": name, "rank": rank,
               "params": layers.n_params(tree), "filter_params": filter_params,
               "param_ratio": dense_params / max(1, filter_params),
               "train_s": train_s}
        try:
            fn = (jax.jit(lambda z: model.forward(tree, z, case.n_modes, F)) if conv is None
                  else jax.jit(lambda z: model.forward(tree, z, case.n_modes, F, conv=conv)))
            tag = f"cprank_{case.label}_{name}_{rank}".replace(" ", "_")
            timing = self.time(fn, (x,), tag, per=case.batch)
            row.update(timing.as_row())
            both = self.errors(fn(x), y)
            row.update(both)
            row["error"] = both[case.loss]
            row["vs_dense"] = (dense_us / timing.mean) if dense_us else 1.0
            row["status"] = "ok"
            print(f"    {name:<8}{str(rank):>5}{filter_params:>9,}"
                  f"{row['param_ratio']:>8.0f}x  {timing}  "
                  f"{case.loss} {row['error']:.4e}  {row['vs_dense']:.3f}x vs dense", flush=True)
        except Exception as exc:
            row["status"] = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
            row.update({k: float("nan") for k in Timing.FIELDS})
            row.update({"error": float("nan"), "h1": float("nan"), "l2": float("nan"),
                        "vs_dense": float("nan")})
            print(f"    {name:<8}{str(rank):>5}{row['status']:>14}  {str(exc)[:50]}", flush=True)
        return row

    def measure(self):
        case = BY_NAME[self.CASE]
        loaded = case.load()
        if loaded is None:
            return []
        _, _, x_test, y_test = loaded
        x = jnp.asarray(np.asarray(x_test[:case.batch], np.float32))
        y = np.asarray(y_test[:case.batch], np.float32)
        print(f"  {case.label}: C={case.channels} m={case.n_modes} B={case.batch} "
              f"| {self.EPOCHS} epochs | ranks {self.RANKS}", flush=True)
        print(f"    {'filter':<8}{'rank':>5}{'filt par':>9}{'vs dense':>9}", flush=True)

        rows = []
        dense_tree = case.weights()
        if dense_tree is None:
            print("    no dense weights yet, cannot place the sweep", flush=True)
            return []
        dense_params = self._filter_params(dense_tree)
        dense_row = self._row(case, "dense", "-", dense_tree, float("nan"), None, x, y,
                              dense_params, None)
        dense_us = dense_row.get("us_mean")
        rows.append(dense_row)

        sep_tree = case.weights("ffno")
        if sep_tree is not None:
            rows.append(self._row(case, "f-fno", "-", sep_tree, float("nan"), separable_conv,
                                  x, y, dense_params, dense_us))

        for rank in self.RANKS:
            tree, train_s = self._weights(case, loaded, rank)
            rows.append(self._row(case, "tfno", rank, tree, train_s, cp_conv, x, y,
                                  dense_params, dense_us))
        return rows


TABLE = CpRank.TABLE


def run():
    return CpRank().run()


if __name__ == "__main__":
    run()
