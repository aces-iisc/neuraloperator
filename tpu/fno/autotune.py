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

    def __init__(self, accel, spec, weights_are_constant=True, max_batch=None):
        self.accel, self.spec = accel, spec
        # Captured weights are folded at compile time, so a relayout or a prebuilt operand costs
        # nothing and buys nothing.  Weights passed in -- always true while training -- are paid
        # for on every call, which is where hoisting and the ADIO layout earn their keep.
        self.weights_are_constant = weights_are_constant
        # nothing can be larger than the split being evaluated
        self.max_batch = max_batch

    # -- individual rules -----------------------------------------------------------
    def capacity(self, dtype="float32"):
        """How many samples the layer's working set keeps on chip at once.

        `working_bytes` counts the field and the buffers the transform holds live beside it,
        not the filter or the DFT matrices, so this is the activation budget rather than a
        true residency figure -- which is why it centres a search instead of capping it.
        """
        cost = LayerCost(self.spec, self.accel, dtype)
        return max(1, self.accel.on_chip_bytes // max(1, cost.working_bytes(1)))

    def batch(self, dtype="float32"):
        """Fill the array if memory allows it, otherwise take what memory allows.

        Step 3 is a batched matmul over the kept modes: the HLO carries both mode indices as
        batch dimensions, so per mode the shape is `B x 2C` against `2C x 2C` and the batch is
        the whole free axis.  The modes do not help fill it.

        The step it moves in is `batch_grain`, NOT the array edge.  They are not the same
        number: every measured optimum on v5e and v6e alike is a multiple of 128, including on
        v6e whose array is 256 wide, and a batch between two multiples loses to both.  Using
        the tile here would also disagree with `candidates`, which steps on the grain -- the
        prediction and the grid it is searched on have to be the same thing.

        Reaching the grain takes the cost per sample from `grain / B` down to 1, and past it
        there is nothing more to win on occupancy, so the prediction is the grain or capacity,
        whichever binds.  Both estimates run low -- Darcy-64 measures fastest at 128 against a
        capacity of 85 -- which is what `candidates` searches around.
        """
        capacity = self.capacity(dtype)
        grain = self.accel.batch_grain or self.accel.tile
        if capacity < grain:
            return capacity, (f"capacity is {capacity} samples, under the {grain} the array "
                              f"consumes at once: memory binds before the free axis fills")
        return grain, (f"capacity holds {capacity} samples, so the batch reaches the "
                       f"{grain}-sample step; past it the free axis is already full")

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
        """The handful of batches worth measuring, on the grid the array makes meaningful.

        `batch` predicts the largest batch that fits, so the search is dense around that
        estimate rather than spread over octaves, and it has to reach a little past it: the
        estimate runs low, and the point just over it is sometimes the fastest.

        Which grid depends on whether the array or memory binds, because only one of them
        quantises the batch.  With room for several tiles the sawtooth above the tile edge is
        real and steep, so every candidate is a multiple of the tile; powers of two are right
        here only while the tile is one.  With room for less than a tile nothing quantises at
        all -- every batch costs the same single pass -- so the cost per sample falls smoothly
        as 1/B and the optimum lands wherever capacity happens to fall.  The candidates there
        are plain integers straddling it, which on Navier-Stokes at single precision is the
        difference between 8 samples and 10.  The tile goes in as well, because filling the
        array can be worth the spill and the model cannot tell in advance whether it is.
        """
        base = self.tune(dtype)
        # The batch steps on `batch_grain`, not on the array edge: measured on both TPU
        # generations, every fastest batch is a multiple of 128 even where the array is 256
        # wide.  Falling back to the tile is only for a target whose grain is not yet measured.
        grain = self.accel.batch_grain or self.accel.tile
        capacity = self.capacity(base.dtype)
        # A split cannot be evaluated on more samples than it holds, so it caps the search as
        # firmly as memory does -- and dropping the candidates above it would leave nothing to
        # measure, so the ceiling itself becomes one.
        ceiling = self.max_batch if self.max_batch else 1 << 30
        bound = min(capacity, ceiling)
        if bound >= grain:
            # The first few multiples.  The reach goes half an estimate past capacity rather
            # than stopping at it, because capacity runs low -- Darcy-64 measures fastest at
            # 256 against a capacity of 113, on a working set 2.25x the on-chip budget -- and a
            # candidate that does not fit costs one failed measurement, while one that is never
            # proposed cannot be found at all.
            values = {k * grain for k in (1, 2, 3, 4) if k * grain <= 1.5 * bound}
            values |= {grain, bound}
        else:
            # Below the grain nothing is quantised, so the ladder is geometric rather than a
            # cluster around `capacity`.  Clustering there assumes the estimate is roughly
            # right, and in this regime it is not: Navier-Stokes in bfloat16 has a capacity of
            # 21 and measures fastest at 4 or below, because halving the operand width halves
            # the matmul but not the transform, and the layer turns bandwidth-bound instead of
            # gaining from a wider batch.  Doubling from one sample costs a handful of cheap
            # measurements and brackets the optimum wherever in the decade it actually sits.
            values, step = {bound}, 1
            while step <= 1.25 * bound:
                values.add(step)
                step *= 2
            # Both hardware widths go in too.  Reaching the array edge can be worth a large
            # spill and the model cannot tell when: Darcy-64 measures fastest at 256 on v6e,
            # more than twice its own capacity, because the relayout its copy bucket pays below
            # the edge -- 427 us per sample at 141, 18 at 256 -- costs more than streaming the
            # overflow does.
            # The optimum repeatedly sits just ABOVE capacity -- Darcy-64 at 128 against 113,
            # Navier-Stokes on v6e at 12 against 10 -- because capacity counts activation bytes
            # and a small spill is cheaper than the samples it buys.  Without these the ladder
            # jumps straight from capacity to the grain and brackets the optimum from one side.
            values |= {max(1, round(bound * f)) for f in (1.25, 1.5)}
            values |= {grain, self.accel.tile}
        values = {min(v, ceiling) for v in values if v >= 1}
        return [Plan(batch=v, dtype=base.dtype, method=base.method,
                     hoist=base.hoist, notes=base.notes) for v in sorted(values)]

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
    def search(self, measure, dtype=None, keep_failures=True, plans=None):
        """Time each candidate and return them ranked, fastest first.

        `measure(plan)` runs the layer under that plan and returns microseconds per sample, or
        raises if the configuration does not fit.  A configuration that runs out of memory is
        information -- it bounds the batch -- so by default it is recorded rather than dropped.
        """
        results = []
        for candidate in (self.candidates(dtype) if plans is None else plans):
            try:
                results.append(Measured(candidate, measure(candidate)))
            except Exception as exc:
                reason = "OOM" if "RESOURCE" in str(exc).upper() else type(exc).__name__
                if keep_failures:
                    results.append(Measured(candidate, float("inf"), status=reason))
        results.sort(key=lambda r: r.us_per_sample)
        return results

    def best(self, measure, dtype=None, refine=True, rounds=3):
        """The fastest candidate that actually ran, refined where the curve is smooth.

        Whether refining helps is a property of the regime, not of the target.  At or above
        `batch_grain` the cost per sample is a sawtooth whose minima ARE the multiples, so
        nothing lives between two rungs and the ladder is already the answer.  Below the grain
        nothing quantises: the optimum sits wherever capacity falls, and the doubling ladder
        steps straight over it -- Navier-Stokes on v6e measures fastest at 12 while the ladder
        offers 10, a 5% loss no amount of re-running the same grid would find.

        The bracket is held to within a factor of two of the winner.  Below the grain the rungs
        are 1, 2, 4, 8, capacity, then the grain itself, so the neighbour above can be ten
        times the winner, and bisecting into that gap spends every probe far from the optimum.
        """
        ranked = self.search(measure, dtype)
        winners = [r for r in ranked if r.ok]
        if not winners:
            raise RuntimeError(f"no candidate ran: {[r.status for r in ranked]}")
        top = winners[0]
        grain = self.accel.batch_grain or self.accel.tile

        if refine and self.capacity(top.plan.dtype) < grain:
            timed = {r.plan.batch: r for r in winners}
            for _ in range(rounds):
                order = sorted(timed)
                winner = min(timed, key=lambda b: timed[b].us_per_sample)
                at = order.index(winner)
                lo = max(order[at - 1] if at else 1, winner // 2, 1)
                hi = min(order[at + 1] if at + 1 < len(order) else winner * 2, winner * 2)
                fresh = [b for b in ((lo + winner) // 2, (winner + hi) // 2)
                         if b >= 1 and b not in timed
                         and (self.max_batch is None or b <= self.max_batch)]
                if not fresh:
                    break
                probes = [Plan(batch=b, dtype=top.plan.dtype, method=top.plan.method,
                               hoist=top.plan.hoist, notes=top.plan.notes) for b in fresh]
                for r in self.search(measure, dtype, plans=probes):
                    if r.ok:
                        timed[r.plan.batch] = r
            top = timed[min(timed, key=lambda b: timed[b].us_per_sample)]
            ranked = sorted(timed.values(), key=lambda r: r.us_per_sample)

        predicted = self.tune(dtype).batch
        top.plan.notes += (f"searched    {sorted(r.plan.batch for r in ranked)}",
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
