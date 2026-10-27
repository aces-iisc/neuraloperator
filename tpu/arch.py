"""Hardware description, cost model and planning for the spectral layer.

Every number the layer tunes against -- how wide a matmul has to be before the unit is full,
how much fits on chip, when a prebuilt operand stops paying for itself -- is a property of the
accelerator, not of the operator.  They were constants spread across autotune.py and fft/core.py
with a TPU v5e baked in.  Here an `Accelerator` owns them, a `LayerCost` derives the byte and
flop counts from one, and an `AutoTune` turns the two into the decisions the layer takes.

Adding a target means adding an `Accelerator`, not editing the layer.
"""
from dataclasses import dataclass, field


# ── the machine ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Accelerator:
    """What the layer needs to know about a matrix engine.

    `tile` is the edge of the systolic array or tensor core: a matmul narrower than this leaves
    lanes idle whatever else is done, so it sets the width the layer tries to reach by stacking
    and blocking, and it sets the radix of a staged transform.

    `batch_grain` is the batch step that measures faster, which is NOT `tile`.  It is its own
    constant because the two come apart: on v5e and v6e alike every measured optimum is a
    multiple of 128, including on v6e whose array is 256 wide, and a batch between two multiples
    loses to both -- 141 samples cost 1.94x of 128 on v6e, and 455 lose to a plain 512.  Deriving
    the step from `tile` makes 384 unreachable on a 256-wide machine, and 384 is the fastest
    batch measured for three of the eight v5e configurations.

    `on_chip_bytes` is the scratch a kernel can keep resident -- VMEM on a TPU, shared memory or
    L2 on a GPU.  An operand larger than this is re-read from main memory on every use, which is
    what decides whether prebuilding one is a saving or a cost.
    """

    name: str
    tile: int                       # systolic / tensor-core edge
    on_chip_bytes: int              # resident scratch per core
    device_bytes: int               # main device memory
    accum_bytes: int = 4            # accumulation width, usually f32
    live_buffers: int = 3           # concurrent activation buffers, calibrated per target
    dtype_bytes: dict = field(default_factory=lambda: {"float32": 4, "bfloat16": 2, "int8": 1})
    transpose_fold_cols: int = None  # stage width above which folding a transpose into the
                                     # matmul pays; None means measure it, do not guess
    batch_grain: int = None          # batch step the hardware actually rewards; None means
                                     # measure it, do not guess

    @property
    def radix(self):
        """A staged transform peels one tile-sized stage at a time."""
        return self.tile

    def bytes_of(self, dtype):
        return self.dtype_bytes[dtype]

    def fits_on_chip(self, nbytes):
        return nbytes <= self.on_chip_bytes

    @staticmethod
    def _si(nbytes):
        for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
            if nbytes >= scale:
                return f"{nbytes / scale:.3g} {unit}"
        return f"{nbytes} B"

    def __str__(self):
        return (f"{self.name}: {self.tile}x{self.tile} tile, "
                f"{self._si(self.on_chip_bytes)} on chip, "
                f"{self._si(self.device_bytes)} device")


TPU_V5E = Accelerator(name="TPU v5e", tile=128, on_chip_bytes=128 * 2**20,
                      device_bytes=16 * 2**30, transpose_fold_cols=64, batch_grain=128)

# Trillium: the systolic array doubles each way, so every tile-edge threshold doubles with it
# and HBM doubles too.  VMEM is NOT assumed to have changed -- `probe_vmem` reads the real
# figure off the device from the compiler's own allocation failure, and that is what this
# records.  `transpose_fold_cols` is a measured crossover and has not been measured here, so it
# stays None and the planner keeps the explicit transpose rather than guessing v5e's number.
TPU_V6E = Accelerator(name="TPU v6e", tile=256, on_chip_bytes=128 * 2**20,
                      device_bytes=32 * 2**30, transpose_fold_cols=None, batch_grain=128)

# A tensor core contracts a 16-wide fragment; shared memory per SM is the resident scratch.
GPU_TENSOR_CORE = Accelerator(name="GPU tensor core", tile=16, on_chip_bytes=228 * 2**10,
                              device_bytes=40 * 2**30, live_buffers=2)

# Smaller array, far less on-chip scratch: the thresholds move by a lot, which is the point.
HEXAGON_32 = Accelerator(name="Qualcomm 32x32", tile=32, on_chip_bytes=8 * 2**20,
                         device_bytes=4 * 2**30, live_buffers=2,
                         dtype_bytes={"float32": 4, "float16": 2, "int8": 1})

TARGETS = {a.name: a for a in (TPU_V5E, TPU_V6E, GPU_TENSOR_CORE, HEXAGON_32)}

# What JAX calls each device, lowercased, against the entry whose constants describe it.  Matched
# on substring because the vendor string varies with runtime version ("TPU v5 lite", "TPU v5e").
DEVICE_KINDS = ((("v6e", "v6 lite", "v6"), TPU_V6E),
                (("v5e", "v5 lite", "v5litepod"), TPU_V5E))


def detect(name=None):
    """The entry describing the hardware this process is attached to.

    The batch rule divides an on-chip budget by a working set and compares the result against a
    tile edge, so both constants have to come from the machine actually running the benchmark.
    Planning a v6e run with v5e constants produces a plausible number that is wrong by 2x on the
    axis the whole decision turns on, and nothing downstream would catch it.

    `name` names a target explicitly, for planning a machine that is not attached.  Otherwise the
    device is asked what it is, and an unrecognised answer raises rather than defaulting: a
    silently wrong target is worse than a refusal.
    """
    if name:
        if name not in TARGETS:
            raise KeyError(f"unknown target {name!r}; have {sorted(TARGETS)}")
        return TARGETS[name]
    import jax
    kind = jax.devices()[0].device_kind
    for needles, accel in DEVICE_KINDS:
        if any(needle in kind.lower() for needle in needles):
            return accel
    raise RuntimeError(f"device_kind {kind!r} matches no entry in arch.py; add one rather than "
                       f"planning this run with another generation's tile and VMEM")


# ── the operator ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """The FNO being run.  `n_modes` is the TOTAL band across both corners."""

    grid: int
    channels: int
    n_modes: int
    dims: int = 2
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1

    @property
    def half(self):
        """Kept column wavenumbers: the input is real, so only one side is stored."""
        return self.n_modes // 2

    @property
    def mode_cells(self):
        """Coefficients that survive truncation, per channel pair.

        One axis is stored half-length because the input is real; the rest keep the full
        band, so a d-dimensional band is `half * n_modes^(d-1)` cells rather than `n_modes^d`.
        """
        return self.half * self.n_modes ** (self.dims - 1)


# ── what it costs ───────────────────────────────────────────────────────────────────
class LayerCost:
    """Byte and flop counts for one spectral layer on one accelerator.

    Everything here is arithmetic on the shapes, not a measurement, so it can be evaluated
    before anything runs -- which is what lets the planner reject a configuration instead of
    discovering it by OOM.
    """

    def __init__(self, spec, accel, dtype="float32"):
        self.spec, self.accel, self.dtype = spec, accel, dtype
        self.width = accel.bytes_of(dtype)

    # -- the filter -----------------------------------------------------------------
    @property
    def filter_bytes(self):
        """Stored filter: (n_modes, n_modes/2, in, out) real and imaginary."""
        s = self.spec
        return 2 * s.channels ** 2 * s.mode_cells * self.width

    @property
    def block_bytes(self):
        """The blocked operand [[re, -im], [im, re]], which is 2x2 the stored filter.

        Quadratic in BOTH channels and modes, so whether prebuilding it pays is a question
        about this number rather than about either one alone.
        """
        s = self.spec
        return 4 * (2 * s.channels) ** 2 * s.mode_cells * self.width // 2

    # -- the activations ------------------------------------------------------------
    def field_bytes(self, batch):
        s = self.spec
        return batch * s.channels * s.grid ** s.dims * self.width

    def working_bytes(self, batch):
        """Field plus the concurrent buffers the transform keeps alive.

        Sized by the ACCUMULATOR, not the operand.  Narrowing the dtype narrows what is fed to
        the matmul, but every stage accumulates and hands on f32, so the buffers that have to
        stay resident do not shrink.  Dividing by the operand width instead says bfloat16 has
        twice the capacity and it does not: Darcy-32 measures fastest at 168 samples in both
        precisions on a 5070, against 170 predicted this way and 341 predicted the other.
        """
        s = self.spec
        return batch * s.channels * s.grid ** s.dims * self.accel.accum_bytes \
            * self.accel.live_buffers

    # -- arithmetic -----------------------------------------------------------------
    def transform_macs(self, batch):
        """Both passes: rows contract the grid, columns contract what the rows left."""
        s = self.spec
        # One pass per axis.  Each contracts that axis from `grid` down to the band, applied
        # across every position the other axes hold -- already banded for the axes done, still
        # full grid for those to come.  The last axis keeps only half the band: real input.
        macs, done = 0, 1
        for axis in range(s.dims):
            kept = s.half if axis == s.dims - 1 else s.n_modes
            macs += kept * s.grid * done * s.grid ** (s.dims - 1 - axis)
            done *= kept
        return batch * s.channels * macs * 2               # forward and inverse

    def mix_macs(self, batch):
        s = self.spec
        return batch * (2 * s.channels) ** 2 * s.mode_cells

    def mix_share(self, batch):
        """Fraction of the layer's arithmetic in the channel mix.

        Small here means a factorised filter cannot matter much; large means it can.
        """
        total = self.transform_macs(batch) + self.mix_macs(batch)
        return self.mix_macs(batch) / total if total else 0.0

    def summary(self, batch):
        si = self.accel._si
        return (f"filter {si(self.filter_bytes)}, block {si(self.block_bytes)}, "
                f"field {si(self.field_bytes(batch))}, "
                f"mix {self.mix_share(batch) * 100:.1f}% of MACs")
