"""The operator plus the plan that runs it: batch, precision, and whether to prebuild."""
import time

import numpy as np
import torch

from ..arch import LayerCost, ModelSpec, detect
from ..fft.core import BF, F
from .autotune import AutoTune
from .model import FNO

DTYPES = {"float32": F, "bfloat16": BF}


class PlannedFNO:
    def __init__(self, spec, accel=None, module=None, plan=None, filter="dense", **kw):
        self.spec = spec
        self.accel = accel or detect()
        self.tuner = AutoTune(spec, self.accel, **kw)
        self.plan = plan or self.tuner.tune()
        self.module = module or FNO(spec.in_channels, spec.out_channels, spec.channels,
                                    spec.n_modes, spec.n_layers, filter,
                                    DTYPES[self.plan.dtype])
        self._ready = False

    @property
    def torch_dtype(self):
        return DTYPES[self.plan.dtype]

    def prepared(self):
        if not self._ready:
            self.module.set_dtype(self.torch_dtype)
            if self.plan.hoist:
                self.module.prebuild(self.torch_dtype)
            self._ready = True
        return self.module

    def to(self, device):
        self.module.to(device)
        self._ready = False
        return self

    @torch.no_grad()
    def forward(self, x):
        return self.prepared()(x)

    def autotune(self, sample, reps=10, device=None):
        """Time the candidate batches on a real input and adopt the fastest.

        The rules give the grid; which point on it wins depends on tiling they do not model,
        so that part is measured.
        """
        device = device or next(self.module.parameters()).device
        sample = torch.as_tensor(np.asarray(sample, np.float32))

        def measure(plan):
            batch = min(plan.batch, sample.shape[0])
            x = sample[:batch].to(device)
            self.module.set_dtype(DTYPES[plan.dtype])
            if plan.hoist:
                self.module.prebuild(DTYPES[plan.dtype])
            with torch.no_grad():
                self.module(x)
                _sync(device)
                times = []
                for _ in range(reps):
                    start = time.perf_counter()
                    self.module(x)
                    _sync(device)
                    times.append((time.perf_counter() - start) * 1e6 / batch)
            return float(np.median(times))

        us, plan = self.tuner.best(measure)
        self.plan, self._ready = plan, False
        return us, plan

    def cost(self):
        return LayerCost(self.spec, self.accel, self.plan.dtype)

    def __repr__(self):
        return (f"<FNO {self.spec.grid}^2 C={self.spec.channels} m={self.spec.n_modes} "
                f"on {self.accel.name} | {self.plan}>")


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
