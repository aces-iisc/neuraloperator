"""
Trains the fast FNO on the three regular-grid benchmarks and reports test rel-L2, wall-clock and
inference cost. Model settings come from the shipped presets; where a preset's \nmodes does not fit
the grid it is clamped to grid//2 and the clamp is printed, because the 2-corner band needs
2*n_modes <= grid.

Epoch budgets are set by DOCS_EPOCHS (default modest) -- these are timed runs on a spot VM, not
converged training, and the output says so.

    PYTHONPATH=~/nop DOCS_EPOCHS=30 ~/nopenv/bin/python -m tpu.experimental.bench.docs_train
"""
import json
import os
import time

import jax
import numpy as np

from ...fno import data, train
from ...fno.layers import n_params
from ...fno.model import forward

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "30"))
OUT = os.environ.get("DOCS_TRAIN_OUT", "/tmp/docs_train")

# (label, loader, grid, preset n_modes, channels, batch, preset name)
CASES = [
    ("Darcy 32^2", lambda: data.load_darcy(ROOT, 32), 32, 16, 24, 8, "FNO_Small2d"),
    ("Burgers 17x16", lambda: data.load_burgers(ROOT), 17, 16, 24, 16, "FNO_Small2d"),
    ("NS 128^2", lambda: data.load_ns(ROOT, 128, n_train=10000, n_test=1000), 128, 64, 64, 8,
     "FNO_Medium2d"),
]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"matmul precision = {jax.config.jax_default_matmul_precision} | epochs = {EPOCHS}",
          flush=True)
    os.makedirs(OUT, exist_ok=True)
    results = []

    for label, loader, grid, preset_modes, channels, batch, preset in CASES:
        try:
            xtr, ytr, xte, yte = loader()
        except Exception as exc:
            print(f"\n{label}: SKIPPED, data unavailable ({type(exc).__name__}: {str(exc)[:70]})",
                  flush=True)
            continue

        # the 2-corner band spans 2*m rows, so it cannot exceed the grid
        m = min(preset_modes, grid // 2)
        clamped = " (clamped from %d, 2*n_modes must fit the grid)" % preset_modes if m != preset_modes else ""
        print(f"\n### {label}  preset={preset}  n_modes={m}{clamped}  channels={channels} "
              f"batch={batch}", flush=True)
        print(f"    train{xtr.shape}  test{xte.shape}", flush=True)

        t0 = time.perf_counter()
        P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=m, channels=channels, n_layers=4,
                                   steps=EPOCHS, batch=batch, name=label, log_every=max(1, EPOCHS // 6),
                                   log_csv=os.path.join(OUT, f"{label.replace(' ', '_').replace('^','')}.csv"),
                                   save=os.path.join(OUT, f"{label.replace(' ', '_').replace('^','')}.npz"))
        infer = train.infer_us_per_sample(P, xte, m)
        row = {"case": label, "preset": preset, "n_modes": m, "preset_n_modes": preset_modes,
               "channels": channels, "batch": batch, "epochs": EPOCHS,
               "train_shape": list(xtr.shape), "test_shape": list(xte.shape),
               "rel_l2": rel, "train_s": secs, "infer_us_per_sample": infer,
               "params": n_params(P), "total_s": time.perf_counter() - t0}
        results.append(row)
        print(f"    rel_l2={rel:.4e}  train={secs:.1f}s  infer={infer:.2f} us/sample  "
              f"params={row['params']:,}", flush=True)

    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # Trained weights are the expensive artefact here -- NS-128 is ~45 min on a v5e -- and a spot
    # VM can vanish at any moment. Tar them next to the summary so one scp retrieves everything,
    # and say loudly where they are, because losing them means retraining from scratch.
    import subprocess
    bundle = "/tmp/docs_train_models.tar.gz"
    subprocess.run(["tar", "czf", bundle, "-C", os.path.dirname(OUT), os.path.basename(OUT)],
                   check=False)
    saved = [f for f in os.listdir(OUT) if f.endswith(".npz")]
    print(f"\n!! FETCH THESE BEFORE DELETING THE VM: {bundle}", flush=True)
    print(f"!! contains {len(saved)} trained model(s): {', '.join(sorted(saved))}", flush=True)
    print(f"\n{'case':<16}{'n_modes':>8}{'chans':>7}{'epochs':>8}{'rel L2':>12}"
          f"{'train s':>10}{'us/sample':>11}{'params':>12}", flush=True)
    for r in results:
        print(f"{r['case']:<16}{r['n_modes']:>8}{r['channels']:>7}{r['epochs']:>8}"
              f"{r['rel_l2']:>12.4e}{r['train_s']:>10.1f}{r['infer_us_per_sample']:>11.2f}"
              f"{r['params']:>12,}", flush=True)
    print("\nTRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
