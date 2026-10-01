"""Choosing how to run the layer: the band, the batch, the precision, the transform.

`AutoTune` owns every decision the layer takes.  Each one is a rule about the accelerator --
how wide the matrix unit is, how much stays on chip, whether the weights are compile-time
constants -- rather than a constant tuned on one machine, so a new target is configured without
re-tuning.  The rules narrow the space to a handful of candidates; `search` times those and
`best` returns the winner, because the last factor of two depends on tiling the rules do not
model.

The band is the one choice that comes from the data instead of the hardware: it is read off the
spectrum of the training targets, which costs one FFT and needs no model.
"""
from dataclasses import dataclass

import numpy as np

from ..arch import LayerCost, ModelSpec, TPU_V5E


@dataclass
class Plan:
    """One configuration of the layer: what to run and how wide."""

    batch: int
    dtype: str = "float32"
    method: str = "reim"            # row transform: reim | coupled | reduced
    stacked: bool = True            # cos and sin in one matmul
    blocked: bool = True            # the 2x2 real block instead of four products
    hoist: bool = False             # prebuild the block rather than rebuild per call
    fold_transpose: bool = False    # make the stage permutation the matmul's write pattern
    notes: tuple = ()

    def why(self):
        return "\n".join(f"  {n}" for n in self.notes)

    def __str__(self):
        flags = [k for k in ("stacked", "blocked", "hoist", "fold_transpose") if getattr(self, k)]
        return (f"batch={self.batch} dtype={self.dtype} method={self.method} "
                f"[{', '.join(flags) or 'plain'}]")


class AutoTune:
    """Chooses how to run the layer on a given accelerator.

    Each decision is a rule about the hardware rather than a constant tuned on one machine, so a
    new target is configured without re-tuning.  The rules narrow the space to a handful of
    candidates; `search` measures those and `best` returns the winner, because the last factor
    of two depends on tiling the rules do not model.
    """

    def __init__(self, accel, spec, weights_are_constant=True):
        self.accel, self.spec = accel, spec
        # Captured weights are folded at compile time, so a relayout or a prebuilt operand costs
        # nothing and buys nothing.  Weights passed in -- always true while training -- are paid
        # for on every call, which is where hoisting and the ADIO layout earn their keep.
        self.weights_are_constant = weights_are_constant

    # -- individual rules -----------------------------------------------------------
    def batch(self, dtype="float32"):
        """Start the sweep where the unit is full, or where memory runs out, whichever binds.

        The contracted axis is 2*channels after blocking.  While that is narrower than the
        tile the free axis is what fills the array, so the batch has to reach the tile edge;
        once it is wider, capacity is the binding constraint instead.
        """
        cost = LayerCost(self.spec, self.accel, dtype)
        per_sample = cost.working_bytes(1)
        capacity = max(1, self.accel.on_chip_bytes // max(1, per_sample))
        if 2 * self.spec.channels < self.accel.tile:
            return self.accel.tile, "narrow tile: batch fills the free axis"
        return capacity, ("wide tile: on-chip capacity binds (an upper bound -- "
                          "it overshoots, the search walks down from here)")

    # Below this the block is cheap enough to rebuild that hoisting is inside run-to-run
    # noise -- measured at ~1 MiB, where the sign does not reproduce between runs.
    HOIST_MIN_BYTES = 5 << 20

    def hoist(self, dtype="float32"):
        """Prebuild the block only when it is worth prebuilding AND stays resident.

        Two thresholds, not one.  A block small enough to rebuild cheaply buys nothing: at about
        1 MiB the measurement does not even reproduce its own sign.  A block larger than on-chip
        scratch is worse than nothing, because prebuilding turns it into a constant streamed
        from main memory on every layer -- 1.84x slower on Navier-Stokes.  It pays in between.
        """
        if self.weights_are_constant:
            return False, "weights fold at compile time: prebuilding materialises a constant"
        block = LayerCost(self.spec, self.accel, dtype).block_bytes
        size = self.accel._si(block)
        if block < self.HOIST_MIN_BYTES:
            return False, (f"block {size} is below the {self.accel._si(self.HOIST_MIN_BYTES)} "
                           f"floor: rebuilding it is already cheap")
        if not self.accel.fits_on_chip(block):
            return False, (f"block {size} exceeds {self.accel._si(self.accel.on_chip_bytes)} "
                           f"on chip: it would stream from memory every layer")
        return True, f"block {size} is large enough to matter and stays resident"

    def fold_transpose(self, n_cols):
        """Fold a stage transpose into the matmul only on stages wide enough to pay.

        The crossover is a property of the memory system, so it is measured per target rather
        than predicted; a target that has not been measured keeps the explicit transpose.
        """
        threshold = self.accel.transpose_fold_cols
        if threshold is None:
            return False, f"{self.accel.name}: crossover not measured, keeping the transpose"
        if n_cols >= threshold:
            return True, f"stage is {n_cols} wide, at or above the measured crossover"
        return False, f"stage is {n_cols} wide, below the {threshold}-column crossover"

    def band(self, targets, tau=0.999, n_sample=64, round_to=8):
        """The smallest band holding `tau` of the radial spectral energy of the targets.

        This is the one decision that comes from the data rather than the machine.  It costs a
        single FFT over a sample of the training targets, so it does not scale with the set and
        needs no trained model.
        """
        y = np.asarray(targets[:n_sample], dtype=np.float64)
        power = np.abs(np.fft.fft2(y, axes=(1, 2))) ** 2
        n = power.shape[1]
        ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij")
        radius = np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
        total = power.sum()
        cutoff = n // 2
        for r in range(1, n // 2 + 1):
            if power[:, radius <= r].sum() / total >= tau:
                cutoff = r
                break
        modes = int(round_to * max(1, round(2 * cutoff / round_to)))   # band reaches radius m/2
        return min(modes, n), f"{tau:.3%} of radial energy inside {modes} modes"

    def energy_curve(self, targets, bands, n_sample=64):
        """Energy fraction inside each band, for reporting how sharp that cutoff is."""
        y = np.asarray(targets[:n_sample], dtype=np.float64)
        power = np.abs(np.fft.fft2(y, axes=(1, 2))) ** 2
        n = power.shape[1]
        ky, kx = np.meshgrid(np.fft.fftfreq(n) * n, np.fft.fftfreq(n) * n, indexing="ij")
        radius = np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)
        total = power.sum()
        return {m: float(power[:, radius <= m // 2].sum() / total) for m in bands}

    def method(self):
        """Which row transform.  Constant in the band an FNO keeps.

        The alternatives trade a smaller matmul for recombination passes, which only wins once
        the band approaches the grid -- a regime the operator never runs in, since the band is
        what makes the layer cheap.
        """
        if self.spec.n_modes >= self.spec.grid:
            return "reduced", "full spectrum: the half-length transform is half the matrix"
        return "reim", "banded: recombination costs more than the smaller matmul saves"

    def dtype(self):
        """Half precision halves activation bytes on a layer bound by them."""
        if "bfloat16" in self.accel.dtype_bytes:
            return "bfloat16", "activations dominate; half width halves the traffic"
        if "float16" in self.accel.dtype_bytes:
            return "float16", "activations dominate; half width halves the traffic"
        return "float32", "no half precision on this target"

    # -- put it together ------------------------------------------------------------
    def tune(self, dtype=None):
        chosen_dtype, dtype_why = (dtype, "requested") if dtype else self.dtype()
        batch, batch_why = self.batch(chosen_dtype)
        hoist, hoist_why = self.hoist(chosen_dtype)
        method, method_why = self.method()
        cost = LayerCost(self.spec, self.accel, chosen_dtype)
        return Plan(batch=batch, dtype=chosen_dtype, method=method, hoist=hoist,
                    stacked=True, blocked=True,
                    notes=(f"target     {self.accel}",
                           f"cost       {cost.summary(batch)}",
                           f"dtype      {chosen_dtype}: {dtype_why}",
                           f"batch      {batch}: {batch_why}",
                           f"method     {method}: {method_why}",
                           f"hoist      {hoist}: {hoist_why}"))

    def candidates(self, dtype=None):
        """The few batches worth measuring: the predicted one and its neighbours.

        The rules get the order of magnitude right; which power of two inside it is fastest
        depends on tiling the rules do not model, so that part is measured.
        """
        base = self.tune(dtype)
        out = []
        for shift in (-2, -1, 0, 1, 2):
            batch = int(base.batch * (2.0 ** shift))
            if batch >= 1 and batch not in [p.batch for p in out]:
                out.append(Plan(batch=batch, dtype=base.dtype, method=base.method,
                                hoist=base.hoist, notes=base.notes))
        return out

    # -- walking around a prediction --------------------------------------------------
    @staticmethod
    def neighbours_mul(value, n=2, lo=1, hi=None):
        """`value` and n powers of two either side.  For the batch, which spans decades."""
        out = []
        for k in range(-n, n + 1):
            v = int(round(value * (2.0 ** k)))
            v = max(lo, v if hi is None else min(v, hi))
            if v not in out:
                out.append(v)
        return sorted(out)

    @staticmethod
    def neighbours_add(value, step=8, n=2, lo=None, hi=None):
        """`value` and n steps either side.  For the band, which moves in fixed increments."""
        lo = step if lo is None else lo
        out = []
        for k in range(-n, n + 1):
            v = value + k * step
            v = max(lo, v if hi is None else min(v, hi))
            if v not in out:
                out.append(v)
        return sorted(out)

    # -- measure what the rules cannot separate --------------------------------------
    def search(self, measure, dtype=None, keep_failures=True):
        """Time each candidate and return them ranked, fastest first.

        `measure(plan)` runs the layer under that plan and returns microseconds per sample, or
        raises if the configuration does not fit.  A configuration that runs out of memory is
        information -- it bounds the batch -- so by default it is recorded rather than dropped.
        """
        results = []
        for candidate in self.candidates(dtype):
            try:
                results.append(Measured(candidate, measure(candidate)))
            except Exception as exc:
                reason = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                if keep_failures:
                    results.append(Measured(candidate, float("inf"), status=reason))
        results.sort(key=lambda r: r.us_per_sample)
        return results

    def best(self, measure, dtype=None):
        """The fastest candidate that actually ran, with the rest kept for the record."""
        ranked = self.search(measure, dtype)
        winners = [r for r in ranked if r.ok]
        if not winners:
            raise RuntimeError(f"no candidate ran: {[r.status for r in ranked]}")
        top = winners[0]
        predicted = self.tune(dtype).batch
        top.plan.notes += (f"searched    {[r.plan.batch for r in ranked]}",
                           f"chose       batch {top.plan.batch} at {top.us_per_sample:.2f} us"
                           f"/sample (rule predicted {predicted})",)
        return top


@dataclass
class Measured:
    """A plan with the time it achieved."""

    plan: Plan
    us_per_sample: float
    status: str = "ok"

    @property
    def ok(self):
        return self.status == "ok" and self.us_per_sample < float("inf")

    def __str__(self):
        if not self.ok:
            return f"{self.plan}  -> {self.status}"
        return f"{self.plan}  -> {self.us_per_sample:.2f} us/sample"
