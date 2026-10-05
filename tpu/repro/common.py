"""Shared objects for the reproduction benches.

Each table is a `Bench` subclass: it declares the table it fills and its columns, and
implements `measure()`.  The base class owns the parts every table needs -- repeated timing
with its spread, the H1 metric, trained-weight lookup and CSV output -- so a measurement is
written once rather than copied between files.

Everything reports relative H1.  The FNO objective is H1, so L2 is not carried here.
"""
import csv
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import BF, F
from ..fno import data, layers, model


class Paths:
    """Where the reproduction reads and writes.  Overridable by environment."""

    data = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
    out = os.path.expanduser(os.environ.get("REPRO_OUT", "~/repro_out"))
    weights = os.path.expanduser(os.environ.get("REPRO_WEIGHTS", "~/repro_out/weights"))
    traces = os.path.expanduser(os.environ.get("REPRO_TRACES", "~/repro_out/traces"))

    @classmethod
    def ensure(cls):
        for d in (cls.out, cls.weights, cls.traces):
            os.makedirs(d, exist_ok=True)

    @classmethod
    def weight_file(cls, label, engine="ours"):
        return os.path.join(cls.weights, f"{label}_{engine}.npz")


class Case:
    """One dataset, with the shape, batch and objective the repository configures it with."""

    def __init__(self, label, loader, channels, n_modes, batch, loss):
        self.label, self._loader = label, loader
        self.channels, self.n_modes, self.batch = channels, n_modes, batch
        # The objective its config in config/ specifies.  A model is reported in the metric it
        # was optimised for, which is also what neuralop does by default; evaluating on anything
        # else selects checkpoints by a criterion the training never saw.
        self.loss = loss

    def load(self):
        """(x_train, y_train, x_test, y_test), or None when the files are not present."""
        try:
            return self._loader()
        except Exception as exc:
            print(f"    {self.label}: data unavailable ({type(exc).__name__})", flush=True)
            return None

    def weights(self, engine="ours"):
        path = Paths.weight_file(self.label, engine)
        if not os.path.exists(path):
            return None
        # F-FNO is a different parameterisation, so its checkpoint has to be poured back into
        # its own skeleton rather than the dense one.
        init = {"ffno": layers.init_fno_separable,
                "tfno": layers.init_fno_cp}.get(engine)
        return model.load_params(path, channels=self.channels, n_modes=self.n_modes, init=init)

    def __repr__(self):
        return f"<Case {self.label} C={self.channels} m={self.n_modes}>"


# loss values follow config/: burgers_config.py is "l2", darcy_config.py and
# navier_stokes_config.py are "h1".
CASES = [
    Case("Burgers", lambda: data.load_burgers(Paths.data), 24, 16, 128, loss="l2"),
    Case("Darcy-32", lambda: data.load_darcy(Paths.data, 32), 24, 16, 128, loss="h1"),
    Case("Darcy-64", lambda: data.load_darcy(Paths.data, 64), 24, 16, 128, loss="h1"),
    Case("NS-128", lambda: data.load_ns(Paths.data, 128, n_train=1000, n_test=200),
         64, 64, 8, loss="h1"),
]
BY_NAME = {c.label: c for c in CASES}


class Timing:
    """The spread of a repeated measurement, not a single number.

    One run is not a measurement: dispatch, clock and neighbours all move it.  Carrying the
    standard deviation and the min/max envelope lets a difference between two rows be read
    against the noise in each.
    """

    FIELDS = ["us_mean", "us_std", "us_min", "us_max", "us_median", "runs"]

    def __init__(self, samples_us, per=1):
        self.samples = np.asarray(samples_us, dtype=np.float64) / per

    mean = property(lambda self: float(self.samples.mean()))
    std = property(lambda self: float(self.samples.std(ddof=1)) if self.samples.size > 1 else 0.0)
    minimum = property(lambda self: float(self.samples.min()))
    maximum = property(lambda self: float(self.samples.max()))
    median = property(lambda self: float(np.median(self.samples)))
    runs = property(lambda self: int(self.samples.size))

    def as_row(self):
        return {"us_mean": self.mean, "us_std": self.std, "us_min": self.minimum,
                "us_max": self.maximum, "us_median": self.median, "runs": self.runs}

    def __str__(self):
        return (f"{self.mean:8.2f} +/- {self.std:6.2f} us "
                f"[{self.minimum:.2f}, {self.maximum:.2f}] n={self.runs}")


class Bench:
    """Base class for one table.

    Subclasses set TABLE and FIELDS and implement `measure()`, returning row dicts.  `run()`
    writes exactly one CSV, named after the table, so the paper-table to file mapping stays
    one-to-one.
    """

    TABLE = None
    FIELDS = []
    REPS = int(os.environ.get("REPRO_REPS", "10"))

    def measure(self):
        raise NotImplementedError

    # -- helpers every table uses ---------------------------------------------------
    def time(self, fn, args, tag, per=1, reps=None):
        """Compile outside the measurement, then repeat it and capture a trace."""
        Paths.ensure()
        reps = reps or self.REPS
        jax.block_until_ready(fn(*args))
        samples = []
        with jax.profiler.trace(os.path.join(Paths.traces, tag)):
            for _ in range(reps):
                start = time.perf_counter()
                jax.block_until_ready(fn(*args))
                samples.append((time.perf_counter() - start) * 1e6)
        return Timing(samples, per=per)

    @staticmethod
    def error(pred, y, loss="h1"):
        """Relative error in `loss`, the metric that model was trained against.

        The loss functions reduce with a SUM over the batch, matching the torch objects the
        training step is meant to reproduce.  A reported error has to be per sample, so the sum
        is divided by the batch -- otherwise every number scales with the batch it was measured
        on and is not comparable to anything.
        """
        pred, y = jnp.asarray(pred), jnp.asarray(y)
        return float(layers.LOSSES[loss](pred, y)) / pred.shape[0]

    @staticmethod
    def errors(pred, y):
        """Both relative metrics from one forward pass, per sample.

        The model is judged on the one it was trained for; the other is carried because the FNO
        literature reports relative L2 and a table with only H1 cannot be placed against it.
        """
        pred, y = jnp.asarray(pred), jnp.asarray(y)
        n = pred.shape[0]
        return {"h1": float(layers.h1_rel(pred, y)) / n,
                "l2": float(layers.lp_rel(pred, y)) / n}

    @staticmethod
    def h1(pred, y):
        """Relative H1 error per sample.  Kept for callers that want it by name."""
        pred, y = jnp.asarray(pred), jnp.asarray(y)
        return float(layers.h1_rel(pred, y)) / pred.shape[0]

    @staticmethod
    def errors_chunked(forward, x, y, batch):
        """Both metrics over a test split too large for one forward.

        Each chunk contributes its own per-sample sum, so the result is the same mean a single
        forward over the whole split would give.
        """
        totals = {"h1": 0.0, "l2": 0.0}
        for start in range(0, x.shape[0], batch):
            pred = forward(jnp.asarray(x[start:start + batch]))
            ys = jnp.asarray(y[start:start + batch])
            totals["h1"] += float(layers.h1_rel(pred, ys))
            totals["l2"] += float(layers.lp_rel(pred, ys))
        return {k: v / x.shape[0] for k, v in totals.items()}

    @staticmethod
    def chain(rows, key="us_mean"):
        """Annotate an ordered ladder with each step's gain over the PREVIOUS step.

        Quoting every step against the baseline makes a step that bought nothing look like it
        bought whatever the steps before it did, so both are reported.
        """
        first = prev = None
        for row in rows:
            value = row.get(key)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                row["gain_prev"] = row["gain_base"] = float("nan")
                continue
            if prev is None:
                first = prev = value
                row["gain_prev"] = row["gain_base"] = 1.0
            else:
                row["gain_prev"], row["gain_base"] = prev / value, first / value
                prev = value
        return rows

    # -- entry point ----------------------------------------------------------------
    def run(self):
        print(f"\n### {self.TABLE}", flush=True)
        rows = self.measure()
        Paths.ensure()
        name = (self.TABLE or self.__class__.__name__).replace(":", "_")
        path = os.path.join(Paths.out, f"{name}.csv")
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"  -> {path}  ({len(rows)} rows)", flush=True)
        return rows
