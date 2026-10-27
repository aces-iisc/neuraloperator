"""Identify the reference implementation's copy traffic by the shapes being copied."""
import sys
import warnings

sys.path.insert(0, "/home/anirudh")
sys.path.insert(0, "/home/anirudh/fno/nop")
warnings.filterwarnings("ignore")

import torch
from torch.profiler import ProfilerActivity, profile

from neuralop.models import FNO as TheirFNO

DEV = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

LEAN = {"use_channel_mlp": False, "channel_mlp_skip": "identity",
        "enforce_hermitian_symmetry": False, "positional_embedding": None}

CASES = [("NS-128", (128, 128), 64, 4, 63)]

for name, shape, ch, batch, m in CASES:
    for tag, kw in (("default", {}), ("four defaults off", LEAN)):
        x = torch.randn(batch, 1, *shape, device=DEV)
        mod = TheirFNO(n_modes=(m, m), in_channels=1, out_channels=1, hidden_channels=ch,
                       n_layers=4, **kw).to(DEV).eval()
        with torch.no_grad():
            for _ in range(5):
                mod(x)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         record_shapes=True) as prof:
                for _ in range(5):
                    mod(x)
                torch.cuda.synchronize()

        print("=" * 90, flush=True)
        print(f"{name}  m={m}  B={batch}  [{tag}]   every op over 50 us, by shape",
              flush=True)
        print("=" * 90, flush=True)
        events = [e for e in prof.key_averages(group_by_input_shape=True)
                  if e.self_device_time_total / 5 > 50]
        for e in sorted(events, key=lambda e: -e.self_device_time_total)[:14]:
            print(f"  {e.self_device_time_total / 5:8.1f} us  x{e.count // 5:3d}  "
                  f"{e.key[:34]:34} {str(e.input_shapes)[:48]}", flush=True)
        del mod, x
        torch.cuda.empty_cache()
        print(flush=True)
