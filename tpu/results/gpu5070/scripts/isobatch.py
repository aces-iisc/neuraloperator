"""Iso-batch, iso-mode inference timing of the reference implementation against ours.

Every row holds the grid, the channel count and the spectral band fixed, and varies only the
batch size and the implementation, so the only thing left to compare is the code that runs.
The band is matched by band, not by `numel`: the reference stores a complex weight, so matching
parameter counts would quietly hand it twice the modes.
"""
import csv
import sys
import warnings

sys.path.insert(0, "/home/anirudh")
sys.path.insert(0, "/home/anirudh/fno/nop")
warnings.filterwarnings("ignore")

import torch

from neuralop.models import FNO as TheirFNO
from torch_fno.fno.model import FNO as OurFNO

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

WEIGHTS = "/home/anirudh/fno"
OUT_CSV = "/home/anirudh/fno/isobatch_ours_vs_theirs.csv"

# (case, grid, channels, our n_modes, their batch, our batch, m as published, m band-matched)
CASES = [("Burgers", (17, 16), 24, 16, 16, 396, 22, 15),
         ("Darcy-32", (32, 32), 24, 16, 16, 128, 22, 15),
         ("Darcy-64", (64, 64), 24, 16, 16, 32, 22, 15),
         ("NS-128", (128, 128), 64, 64, 8, 4, 78, 63)]

BATCHES = {"Burgers": [16, 64, 128, 256, 396, 512],
           "Darcy-32": [16, 32, 64, 128, 256],
           "Darcy-64": [8, 16, 32, 64, 128],
           "NS-128": [1, 2, 4, 8, 16]}

LEAN = {"use_channel_mlp": False, "channel_mlp_skip": "identity",
        "enforce_hermitian_symmetry": False, "positional_embedding": None}

OURS_LABEL = "ours torch"


def timed(f, x, reps=20, warm=8):
    with torch.no_grad():
        for _ in range(warm):
            f(x)
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        best = float("inf")
        for _ in range(reps):
            begin.record()
            f(x)
            end.record()
            torch.cuda.synchronize()
            best = min(best, begin.elapsed_time(end) * 1e3)
    return best


def real_params(mod):
    """Float count, so a complex weight is not undercounted by half."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in mod.parameters())


def native_model(ours):
    """The same maths written straight on `torch.fft.rfft2` and two einsums."""
    from torch_fno.fno.layers import add_grid, gelu

    def run(x):
        half = ours.n_modes // 2
        x = ours.lift(add_grid(x))
        for block in ours.blocks:
            n, c, h, w = x.shape
            weight = torch.complex(block.spectral.w_re, block.spectral.w_im)
            xf = torch.fft.rfft2(x, dim=(-2, -1))
            out = torch.zeros(n, weight.shape[3], h, w // 2 + 1, dtype=xf.dtype,
                              device=x.device)
            out[:, :, :half, :half] = torch.einsum("adio,niad->noad", weight[:half],
                                                   xf[:, :, :half, :half])
            out[:, :, h - half:, :half] = torch.einsum("adio,niad->noad", weight[half:],
                                                       xf[:, :, h - half:, :half])
            spectral = torch.fft.irfft2(out, s=(h, w), dim=(-2, -1))
            x = gelu(spectral + block.skip(x))
        return ours.project(x)
    return run


def build_ours(name, channels, modes):
    mod = OurFNO(1, 1, channels, modes, 4)
    mod.load_state_dict(torch.load(f"{WEIGHTS}/{name}_state.pt", map_location="cpu"))
    mod = mod.to(DEV).eval()
    for block in mod.blocks:
        block.spectral.layout = "mode"
    mod.prebuild(torch.float32)
    return mod


def build_theirs(channels, m, **kw):
    return TheirFNO(n_modes=(m, m), in_channels=1, out_channels=1, hidden_channels=channels,
                    n_layers=4, **kw).to(DEV).eval()


def batch_kind(batch, their_batch, our_batch):
    if batch == their_batch and batch == our_batch:
        return "theirs+ours"
    if batch == their_batch:
        return "theirs"
    if batch == our_batch:
        return "ours"
    return "sweep"


rows = []
for name, shape, channels, modes, their_batch, our_batch, m_pub, m_fair in CASES:
    print(f"\n{'=' * 90}\n{name}  grid {shape}  C={channels}  our m={modes}  "
          f"their m={m_pub} published / {m_fair} band-matched\n{'=' * 90}", flush=True)

    ours = build_ours(name, channels, modes)
    their_fair = build_theirs(channels, m_fair)
    their_pub = build_theirs(channels, m_pub)
    their_lean = build_theirs(channels, m_fair, **LEAN)

    entries = [
        (OURS_LABEL, ours, modes, real_params(ours)),
        (f"their torch, m={m_fair} (band-matched)", their_fair, m_fair,
         real_params(their_fair)),
        (f"their torch, m={m_pub} (as published)", their_pub, m_pub, real_params(their_pub)),
        (f"their torch, m={m_fair} + four defaults off", their_lean, m_fair,
         real_params(their_lean)),
        ("native torch (rfft2)", native_model(ours), modes, real_params(ours)),
    ]

    batches = sorted(set(BATCHES[name]) | {their_batch, our_batch})
    for batch in batches:
        kind = batch_kind(batch, their_batch, our_batch)
        print(f"\n  B={batch}  ({kind})", flush=True)
        try:
            x = torch.randn(batch, 1, *shape, device=DEV)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            for label, _, m, params in entries:
                rows.append({"case": name, "grid": f"{shape[0]}x{shape[1]}", "batch": batch,
                             "batch_kind": kind, "impl": label, "m": m,
                             "real_params": params, "us_per_sample": "",
                             "x_vs_ours": "", "status": "oom building input"})
            print("    input alone is out of memory", flush=True)
            continue

        here = []
        for label, f, m, params in entries:
            status, us = "ok", ""
            try:
                us = timed(f, x) / batch
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                status = "oom"
            except RuntimeError as err:
                torch.cuda.empty_cache()
                status = f"error: {str(err).splitlines()[0][:80]}"
            shown = f"{us:9.2f}" if status == "ok" else f"{status:>9}"
            print(f"    {label:42} {shown} us/sa   real params {params:,}", flush=True)
            here.append({"case": name, "grid": f"{shape[0]}x{shape[1]}", "batch": batch,
                         "batch_kind": kind, "impl": label, "m": m, "real_params": params,
                         "us_per_sample": us, "x_vs_ours": "", "status": status})

        base = next((r["us_per_sample"] for r in here
                     if r["impl"] == OURS_LABEL and r["status"] == "ok"), None)
        if base:
            for r in here:
                if r["status"] == "ok":
                    r["x_vs_ours"] = r["us_per_sample"] / base
        rows.extend(here)
        del x
        torch.cuda.empty_cache()

    del ours, their_fair, their_pub, their_lean, entries
    torch.cuda.empty_cache()

keys = ["case", "grid", "batch", "batch_kind", "impl", "m", "real_params", "us_per_sample",
        "x_vs_ours", "status"]
with open(OUT_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(f"\nwrote {OUT_CSV}", flush=True)
