"""Hardware constants the layer is planned against."""
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Accelerator:
    name: str
    tile: int                        # systolic / tensor-core edge
    on_chip_bytes: int               # resident scratch per core (SMEM + L1 on a GPU)
    device_bytes: int
    accum_bytes: int = 4
    live_buffers: int = 3
    dtype_bytes: dict = field(default_factory=lambda: {
        "float32": 4, "bfloat16": 2, "float16": 2, "int8": 1})
    transpose_fold_cols: int = None   # measured crossover; None means do not guess
    batch_grain: int = None           # batch step the hardware rewards; None means measure it

    @property
    def radix(self):
        return self.tile

    def bytes_of(self, dtype):
        return self.dtype_bytes[str(dtype).replace("torch.", "")]

    def fits_on_chip(self, nbytes):
        return nbytes <= self.on_chip_bytes

    @staticmethod
    def _si(nbytes):
        for unit, scale in (("GiB", 2 ** 30), ("MiB", 2 ** 20), ("KiB", 2 ** 10)):
            if nbytes >= scale:
                return f"{nbytes / scale:.3g} {unit}"
        return f"{nbytes} B"

    def __str__(self):
        return (f"{self.name}: {self.tile}x{self.tile} tile, {self._si(self.on_chip_bytes)} "
                f"on chip, {self._si(self.device_bytes)} device")


# A tensor core contracts a 16-wide fragment.
#
# `on_chip_bytes` is L2, NOT shared memory.  On a TPU the activations themselves live in VMEM,
# so the batch is bounded by it; on a GPU they live in HBM and shared memory is per-block
# tiling scratch a kernel carves up, not a residency budget for the working set.  Dividing the
# working set by SMEM says one sample fits and predicts a batch of 1.  L2 is the level that
# actually holds a reused working set, so it is the one the batch rule is about.
#
# These are datasheet figures.  `batch_grain` is deliberately None: it is a measured crossover
# (128 on both TPU generations) and nobody has measured it here, so the planner falls back to
# the tile rather than inheriting a number from another vendor.
A100 = Accelerator(name="NVIDIA A100", tile=16, on_chip_bytes=40 * 2 ** 20,
                   device_bytes=40 * 2 ** 30)
H100 = Accelerator(name="NVIDIA H100", tile=16, on_chip_bytes=50 * 2 ** 20,
                   device_bytes=80 * 2 ** 30)
RTX_4090 = Accelerator(name="NVIDIA RTX 4090", tile=16, on_chip_bytes=72 * 2 ** 20,
                       device_bytes=24 * 2 ** 30)
# Blackwell, sm_120.  48 SMs, L2 and memory read off the device rather than a datasheet.
# live_buffers stays at the default 3: it counts what the transform holds, not a vendor
# property, and 3 is what the measured optimum agrees with here (Darcy-32 capacity 170
# against a measured 168; 2 would say 256).
RTX_5070 = Accelerator(name="NVIDIA GeForce RTX 5070", tile=16, on_chip_bytes=48 * 2 ** 20,
                       device_bytes=11769 * 2 ** 20)
CPU = Accelerator(name="CPU", tile=8, on_chip_bytes=32 * 2 ** 20,
                  device_bytes=64 * 2 ** 30)

TARGETS = {a.name: a for a in (A100, H100, RTX_4090, RTX_5070, CPU)}

# Matched on substring because the vendor string carries board and memory variants.
DEVICE_NAMES = ((("a100",), A100), (("h100",), H100),
                (("4090",), RTX_4090), (("5070",), RTX_5070))


def detect(name=None):
    """The entry describing the attached device; refuses rather than guessing."""
    if name:
        if name not in TARGETS:
            raise KeyError(f"unknown target {name!r}; have {sorted(TARGETS)}")
        return TARGETS[name]
    if not torch.cuda.is_available():
        return CPU
    kind = torch.cuda.get_device_name(0).lower()
    for needles, accel in DEVICE_NAMES:
        if any(n in kind for n in needles):
            return accel
    raise RuntimeError(f"device {kind!r} matches no entry in arch.py; add one rather than "
                       f"planning this run with another part's tile and scratch")


@dataclass(frozen=True)
class ModelSpec:
    """The FNO being run. `n_modes` is the TOTAL band across both corners."""

    grid: int
    channels: int
    n_modes: int
    dims: int = 2
    n_layers: int = 4
    in_channels: int = 1
    out_channels: int = 1

    @property
    def half(self):
        return self.n_modes // 2

    @property
    def mode_cells(self):
        return self.half * self.n_modes ** (self.dims - 1)


class LayerCost:
    """Byte and flop counts for one spectral layer, from the shapes alone."""

    def __init__(self, spec, accel, dtype="float32"):
        self.spec, self.accel, self.dtype = spec, accel, dtype
        self.width = accel.bytes_of(dtype)

    @property
    def filter_bytes(self):
        s = self.spec
        return 2 * s.channels ** 2 * s.mode_cells * self.width

    @property
    def block_bytes(self):
        s = self.spec
        return 4 * (2 * s.channels) ** 2 * s.mode_cells * self.width // 2

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

    def transform_macs(self, batch):
        s = self.spec
        macs, done = 0, 1
        for axis in range(s.dims):
            kept = s.half if axis == s.dims - 1 else s.n_modes
            macs += kept * s.grid * done * s.grid ** (s.dims - 1 - axis)
            done *= kept
        return batch * s.channels * macs * 2

    def mix_macs(self, batch):
        s = self.spec
        return batch * (2 * s.channels) ** 2 * s.mode_cells

    def summary(self, batch):
        return (f"filter {self.accel._si(self.filter_bytes)}, "
                f"block {self.accel._si(self.block_bytes)}, "
                f"working {self.accel._si(self.working_bytes(batch))}, "
                f"{(self.transform_macs(batch) + self.mix_macs(batch)) / 1e6:.1f} MMAC")
