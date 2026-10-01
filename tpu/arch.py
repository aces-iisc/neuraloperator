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
                      device_bytes=16 * 2**30, transpose_fold_cols=64)

# A tensor core contracts a 16-wide fragment; shared memory per SM is the resident scratch.
GPU_TENSOR_CORE = Accelerator(name="GPU tensor core", tile=16, on_chip_bytes=228 * 2**10,
                              device_bytes=40 * 2**30, live_buffers=2)

# Smaller array, far less on-chip scratch: the thresholds move by a lot, which is the point.
HEXAGON_32 = Accelerator(name="Qualcomm 32x32", tile=32, on_chip_bytes=8 * 2**20,
                         device_bytes=4 * 2**30, live_buffers=2,
                         dtype_bytes={"float32": 4, "float16": 2, "int8": 1})

TARGETS = {a.name: a for a in (TPU_V5E, GPU_TENSOR_CORE, HEXAGON_32)}


# ── the operator ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ModelSpec:
    """The FNO being run.  `n_modes` is the TOTAL band across both corners."""

    grid: int
    channels: int
    n_modes: int
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1

    @property
    def half(self):
        """Kept column wavenumbers: the input is real, so only one side is stored."""
        return self.n_modes // 2

    @property
    def mode_cells(self):
        """Coefficients that survive truncation, per channel pair."""
        return self.n_modes * self.half


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
        return batch * s.channels * s.grid ** 2 * self.width

    def working_bytes(self, batch):
        """Field plus the concurrent buffers the transform keeps alive."""
        return self.field_bytes(batch) * self.accel.live_buffers

    # -- arithmetic -----------------------------------------------------------------
    def transform_macs(self, batch):
        """Both passes: rows contract the grid, columns contract what the rows left."""
        s = self.spec
        rows = s.n_modes * s.grid * s.grid
        cols = s.n_modes * s.half * s.grid
        return batch * s.channels * (rows + cols) * 2      # forward and inverse

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
