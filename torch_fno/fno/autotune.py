"""Pick the band, the batch, the precision, the transform and whether to prebuild the filter.

Four of the five come from the hardware description and one does not: the band is a property of
the data, so it is the only decision that survives a change of machine unchanged, and it is the
only one that cannot be revised after training.

The batch is the one decision worth measuring rather than deriving. Step 3 is a batched matmul
over the kept modes -- the mode indices are batch dimensions -- so the batch is the whole free
axis of the channel contraction, and the array is consumed in whole `batch_grain`-wide passes.
A batch between two multiples of the grain pays for the next pass and carries nothing extra
with it, so the optimum always sits on a multiple and never between two.
"""
from dataclasses import dataclass

import numpy as np

from ..arch import LayerCost, detect


@dataclass(frozen=True)
class Plan:
    batch: int
    dtype: str
    hoist: bool
    notes: tuple = ()

    def __str__(self):
        return f"batch={self.batch} {self.dtype}{' hoisted' if self.hoist else ''}"


class AutoTune:
    HOIST_MIN_BYTES = 5 << 20

    def __init__(self, spec, accel=None, weights_are_constant=False, max_batch=None):
        self.spec = spec
        self.accel = accel or detect()
        self.weights_are_constant = weights_are_constant
        self.max_batch = max_batch

    def capacity(self, dtype="float32"):
        """Samples whose working set stays on chip. An activation budget, not a hard ceiling."""
        cost = LayerCost(self.spec, self.accel, dtype)
        return max(1, self.accel.on_chip_bytes // max(1, cost.working_bytes(1)))

    def dtype(self):
        if self.accel.name == "CPU":
            return "float32", "no tensor cores: half precision is emulated"
        return "bfloat16", "activations dominate; half width halves the traffic"

    def batch(self, dtype="float32"):
        grain = self.accel.batch_grain or self.accel.tile
        capacity = self.capacity(dtype)
        if capacity < grain:
            return capacity, (f"capacity is {capacity} samples, under the {grain} the array "
                              f"consumes at once: memory binds before the free axis fills")
        return (capacity // grain) * grain, (f"largest multiple of {grain} inside a capacity "
                                             f"of {capacity}")

    def hoist(self, dtype="float32"):
        """Prebuild the block only when it is worth building AND stays resident."""
        if self.weights_are_constant:
            return False, "weights fold at compile time: prebuilding materialises a constant"
        block = LayerCost(self.spec, self.accel, dtype).block_bytes
        size = self.accel._si(block)
        if block < self.HOIST_MIN_BYTES:
            return False, f"block {size} is below the rebuild-is-cheap floor"
        if not self.accel.fits_on_chip(block):
            return False, f"block {size} exceeds on-chip scratch: it would stream every layer"
        return True, f"block {size} is large enough to matter and stays resident"

    @staticmethod
    def _radial_power(targets, n_sample):
        """Radial wavenumber of every bin, and the energy at each, for a sample of the targets.

        Two things differ from the TPU spelling this is ported from, both of them about shape
        rather than hardware. The loaders here hand back (N, C, H, W) where that tree's hand
        back (N, H, W), so the channel axis is folded into the sample axis rather than
        transformed over: a channel is not a spatial direction and a DFT along it would be
        meaningless. And the two spatial axes are taken separately instead of assuming a square
        grid, because Burgers is 17 by 16 -- one axis is time and has an endpoint. The radius is
        in signed bin indices either way, so for a square grid this is the same computation.

        The band it feeds is bounded by the shorter axis: a band is kept on both axes, so the
        smaller Nyquist is the one that binds.
        """
        y = np.asarray(targets[:n_sample], dtype=np.float64)
        y = y.reshape(-1, *y.shape[-2:])
        power = np.abs(np.fft.fft2(y, axes=(1, 2))) ** 2
        rows, cols = power.shape[1], power.shape[2]
        ky, kx = np.meshgrid(np.fft.fftfreq(rows) * rows, np.fft.fftfreq(cols) * cols,
                             indexing="ij")
        return power, min(rows, cols), np.round(np.sqrt(kx ** 2 + ky ** 2)).astype(int)

    def band(self, targets, tau=0.999, n_sample=64, round_to=8):
        """The smallest band holding `tau` of the radial spectral energy of the targets.

        This is the one decision that comes from the data rather than the machine. It costs a
        single FFT over a sample of the training targets, so it does not scale with the set and
        needs no trained model -- which is what lets it run before the training it decides.
        """
        power, n, radius = self._radial_power(targets, n_sample)
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
        power, _, radius = self._radial_power(targets, n_sample)
        total = power.sum()
        return {m: float(power[:, radius <= m // 2].sum() / total) for m in bands}

    def method(self):
        """Which row transform. Constant in the band an FNO keeps.

        The alternatives trade a smaller matmul for recombination passes, which only wins once
        the band approaches the grid -- a regime the operator never runs in, since the band is
        what makes the layer cheap.
        """
        if self.spec.n_modes >= self.spec.grid:
            return "reduced", "full spectrum: the half-length transform is half the matrix"
        return "reim", "banded: recombination costs more than the smaller matmul saves"

    def tune(self, dtype=None):
        chosen, why_dtype = (dtype, "requested") if dtype else self.dtype()
        batch, why_batch = self.batch(chosen)
        hoist, why_hoist = self.hoist(chosen)
        engine, why_engine = self.method()
        cost = LayerCost(self.spec, self.accel, chosen)
        return Plan(batch=batch, dtype=chosen, hoist=hoist,
                    notes=(f"target {self.accel}",
                           f"cost   {cost.summary(batch)}",
                           f"dtype  {chosen}: {why_dtype}",
                           f"batch  {batch}: {why_batch}",
                           f"hoist  {hoist}: {why_hoist}",
                           f"engine {engine}: {why_engine}"))

    def candidates(self, dtype=None):
        """The few batches worth measuring, on the grid the array makes meaningful.

        Above the grain only its multiples are worth trying, and the sawtooth between them is
        steep. Below it nothing quantises, so the ladder is geometric from one sample rather
        than a cluster around `capacity` -- that estimate can be several times too high when
        the layer turns bandwidth-bound, and a cluster could not reach down to the optimum.
        """
        base = self.tune(dtype)
        grain = self.accel.batch_grain
        capacity = self.capacity(base.dtype)
        ceiling = self.max_batch if self.max_batch else 1 << 30
        bound = min(capacity, ceiling)

        if grain and bound >= grain:
            # A measured grain means the array is consumed in whole passes and a batch between
            # two multiples pays for the next one, so only the multiples are worth trying.
            values = {k * grain for k in (1, 2, 3, 4) if k * grain <= 1.5 * bound}
            values |= {grain, bound}
        elif grain:
            values, step = {bound, grain}, 1
            while step <= 1.25 * bound:
                values.add(step)
                step *= 2
        else:
            # No grain has been measured on this target, so do not invent one.  The tile is the
            # tensor-core fragment, not a batch quantum: a 5070 shows no sawtooth at all (worst
            # local bump 1.017x against 2.23x on a TPU) because the batch spreads over thread
            # blocks rather than filling one fixed array pass.  Search geometrically around the
            # capacity estimate instead, which is what actually locates the knee there.
            values = {max(1, int(round(bound * f))) for f in (0.25, 0.5, 0.75, 1.0, 1.25)}
            values.add(1)
        values = {min(v, ceiling) for v in values if v >= 1}
        return [Plan(batch=v, dtype=base.dtype, hoist=base.hoist, notes=base.notes)
                for v in sorted(values)]

    def search(self, measure, dtype=None, plans=None):
        """Time every candidate and return them ranked, fastest first."""
        out = []
        for plan in (plans if plans is not None else self.candidates(dtype)):
            try:
                out.append((float(measure(plan)), plan))
            except RuntimeError as exc:                 # an OOM candidate simply drops out
                out.append((float("inf"), plan))
                if "out of memory" not in str(exc).lower():
                    raise
        return sorted(out, key=lambda r: r[0])

    def best(self, measure, dtype=None, refine=True, rounds=3):
        """The fastest batch, refined between the ladder rungs where that can help.

        Whether refinement is worth anything depends on the hardware, not on the model.  With a
        measured `batch_grain` the cost per sample is a sawtooth: the minima sit exactly on the
        multiples and every batch between two of them pays for the next pass, so there is
        nothing to find in the gaps and the ladder is already the answer.  Without one the
        curve is smooth -- a 5070 shows a worst local bump of 1.017x against 2.23x on a TPU --
        so the optimum falls wherever capacity happens to land, which is between rungs more
        often than not.  Bisecting the winner's bracket costs a few measurements and closes a
        gap the ladder cannot: 422 samples on Burgers, against 302 from the ladder alone.
        """
        ranked = self.search(measure, dtype)
        grain = self.accel.batch_grain
        base_dtype = ranked[0][1].dtype
        # Refine only where the curve is actually smooth.  A grain makes the cost per sample a
        # sawtooth ABOVE it -- the minima are the multiples and nothing lives between two of
        # them -- but below it nothing quantises at all, so the optimum lands wherever capacity
        # falls and a doubling ladder steps over it.  Gating on "has a grain" rather than "is
        # in the quantised regime" is what cost Navier-Stokes 5% on v6e: capacity 10 against a
        # grain of 128, a smooth curve, and refinement switched off.
        if not refine or (grain and self.capacity(base_dtype) >= grain):
            return ranked[0]

        timed = {plan.batch: us for us, plan in ranked}
        base = ranked[0][1]
        for _ in range(rounds):
            order = sorted(timed)
            winner = min(timed, key=timed.get)
            at = order.index(winner)
            # Bound the bracket to within a factor of two of the winner.  The ladder leaves
            # wide gaps -- below the grain its rungs are 1, 2, 4, 8, capacity, then the grain
            # itself -- so the neighbour above can be ten times the winner, and bisecting into
            # that gap spends every probe far from the optimum instead of closing on it.
            lo = max(order[at - 1] if at else 1, winner // 2, 1)
            hi = min(order[at + 1] if at + 1 < len(order) else winner * 2, winner * 2)
            fresh = [b for b in ((lo + winner) // 2, (winner + hi) // 2)
                     if b >= 1 and b not in timed
                     and (self.max_batch is None or b <= self.max_batch)]
            if not fresh:
                break
            probes = [Plan(batch=b, dtype=base.dtype, hoist=base.hoist, notes=base.notes)
                      for b in fresh]
            timed.update({plan.batch: us for us, plan in
                          self.search(measure, dtype, plans=probes)})
        winner = min(timed, key=timed.get)
        return timed[winner], Plan(batch=winner, dtype=base.dtype, hoist=base.hoist,
                                   notes=base.notes)
