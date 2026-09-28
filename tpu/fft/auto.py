"""Picks the transform engine and backend for a given size, from measured device time."""
from .core import F
from .rfft1d import rfft_direct, rfft_half, rfft_low_modes

BACKEND_XLA = "xla"
BACKEND_PALLAS = "pallas"

# (engine, backend) per size band, read straight off the table above.
FULL_PLAN = [
    (128, "reim", BACKEND_XLA),        # 1.6 vs 1.9 for Pallas
    (256, "reduced", BACKEND_PALLAS),  # 3.7, the best measured at this size
    (512, "direct", BACKEND_XLA),      # 12.0, the full DFT still wins here
    (2048, "reim", BACKEND_PALLAS),    # 16.0 and 30.8, Pallas ahead by 1.14x
]
FULL_FALLBACK = ("reim", BACKEND_XLA)  # N > 2048: Pallas OOMs, XLA is faster anyway

PALLAS_MAX_N = 2048     # above this a Pallas kernel exhausts the default scoped VMEM budget


def plan(n, low_modes=None):
    """Return (engine, backend) for a length-n transform. low_modes set means only that many
    bins are wanted, which is always the XLA partial path."""
    if low_modes is not None:
        return "partial", BACKEND_XLA
    for limit, engine, backend in FULL_PLAN:
        if n <= limit:
            return engine, backend
    return FULL_FALLBACK


def describe(n, low_modes=None):
    engine, backend = plan(n, low_modes)
    return f"N={n} -> {backend} {engine}"


def auto_rfft(signal, low_modes=None, dtype=F, allow_pallas=True):
    """Transform `signal` (n, batch) with whichever engine measured fastest at that n.

    Falls back to the XLA equivalent if a Pallas kernel fails to compile or runs out of VMEM, so
    a bad plan costs time rather than correctness."""
    n = signal.shape[0]
    engine, backend = plan(n, low_modes)

    if low_modes is not None:
        return rfft_low_modes(signal, low_modes, dtype)

    if backend == BACKEND_PALLAS and allow_pallas and n <= PALLAS_MAX_N:
        try:
            return _pallas(engine, signal, dtype)
        except Exception:
            pass                       # OOM or a missing Mosaic primitive: use XLA instead
    return _xla(engine, signal, dtype)


def _xla(engine, signal, dtype):
    if engine == "direct":
        return rfft_direct(signal, True, dtype)
    return rfft_half(signal, dtype)    # reim, and the fallback for reduced/coupled


def _pallas(engine, signal, dtype):
    from ..experimental.fft.pallas1d import pallas_half, pallas_reduced
    batch = signal.shape[1]
    if engine == "reduced":
        return pallas_reduced(signal, batch, False, dtype)
    return pallas_half(signal, batch, False, dtype)
