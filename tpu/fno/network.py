"""The operator as one object: a spec, an accelerator, and the plan that connects them.

`FNO` holds the weights and runs the layer under a `Plan`.  The plan says how wide to batch,
what precision to carry, whether to prebuild the channel block -- decisions `AutoTune` makes
from the accelerator rather than from constants tuned on one machine.

The transform is an `FFT`, so changing target changes the radix and the thresholds without the
operator being edited.
"""
import jax
import jax.numpy as jnp
import numpy as np

from ..arch import TPU_V5E, LayerCost, ModelSpec
from ..fft.core import BF, F
from ..fft.transform import FFT
from . import layers, model
from .autotune import AutoTune


class FNO:
    """A Fourier neural operator bound to one accelerator and one plan."""

    DTYPES = {"float32": F, "bfloat16": BF}

    def __init__(self, spec, accel=TPU_V5E, params=None, plan=None, seed=0):
        self.spec, self.accel = spec, accel
        self.tuner = AutoTune(accel, spec)
        self.plan = plan or self.tuner.tune()
        self.params = params if params is not None else self._init(seed)
        self._prepared = None

    # -- construction ----------------------------------------------------------------
    def _init(self, seed):
        return layers.init_fno(jax.random.PRNGKey(seed), self.spec.in_channels,
                               self.spec.out_channels, self.spec.channels,
                               self.spec.n_modes, self.spec.n_layers)

    @classmethod
    def load(cls, path, spec, accel=TPU_V5E, **kw):
        params = model.load_params(path, channels=spec.channels, n_modes=spec.n_modes,
                                   n_layers=spec.n_layers)
        return cls(spec, accel, params=params, **kw)

    def save(self, path):
        return model.save_params(self.params, path)

    # -- the weights the plan asks for -------------------------------------------------
    @property
    def jax_dtype(self):
        return self.DTYPES[self.plan.dtype]

    def prepared(self):
        """Weights in the form the plan wants, built once.

        Prebuilding the channel block is only a saving when it stays resident and the weights
        are not compile-time constants; `AutoTune.hoist` decides, and this honours it.
        """
        if self._prepared is None:
            self._prepared = (layers.prebuild_blocks(self.params, self.jax_dtype)
                              if self.plan.hoist else self.params)
        return self._prepared

    # -- running it --------------------------------------------------------------------
    def forward(self, x):
        return model.forward(self.prepared(), x, self.spec.n_modes, self.jax_dtype)

    def jitted(self, weights_as_argument=False):
        """A compiled forward.

        Capturing the weights lets XLA fold them, which removes the filter relayout at compile
        time but also makes a prebuilt block a materialised constant re-read every call.  Pass
        them in instead -- as training always does -- and both are paid for per call, which is
        where the layout and the hoist earn their keep.
        """
        n_modes, dtype = self.spec.n_modes, self.jax_dtype
        if weights_as_argument:
            return jax.jit(lambda p, z: model.forward(p, z, n_modes, dtype))
        weights = self.prepared()
        return jax.jit(lambda z: model.forward(weights, z, n_modes, dtype))

    # -- tuning --------------------------------------------------------------------------
    def autotune(self, sample, reps=10):
        """Time the candidate batches on a real input and adopt the fastest.

        The rules get the order of magnitude; which power of two inside it wins depends on
        tiling they do not model, so that part is measured.
        """
        import time

        def measure(plan):
            batch = min(plan.batch, sample.shape[0])
            x = jnp.asarray(np.asarray(sample[:batch], np.float32))
            fn = jax.jit(lambda z: model.forward(self.params, z, self.spec.n_modes,
                                                 self.DTYPES[plan.dtype]))
            jax.block_until_ready(fn(x))
            samples = []
            for _ in range(reps):
                start = time.perf_counter()
                jax.block_until_ready(fn(x))
                samples.append((time.perf_counter() - start) * 1e6 / batch)
            return float(np.median(samples))

        best = self.tuner.best(measure)
        self.plan, self._prepared = best.plan, None
        return best

    # -- what it costs --------------------------------------------------------------------
    def cost(self):
        return LayerCost(self.spec, self.accel, self.plan.dtype)

    def n_params(self):
        return layers.n_params(self.params)

    def __repr__(self):
        return (f"<FNO {self.spec.grid}^2 C={self.spec.channels} m={self.spec.n_modes} "
                f"on {self.accel.name} | {self.plan}>")
