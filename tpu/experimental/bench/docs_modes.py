"""
Sweeps n_modes and records what it costs and what it buys: final test rel-L2 after a fixed
training budget, and inference time per sample. Precision highest throughout.

The swept value is PER CORNER, so the total band is twice it: the column printed as 8 means
num_modes = 16. The ceiling is grid/2 because the two corners together need 2*m rows.

    PYTHONPATH=~/nop DOCS_EPOCHS=30 ~/nopenv/bin/python -m tpu.experimental.bench.docs_modes
"""
import csv, json, os, subprocess, time
import jax
import numpy as np

from ...fno import data, train
from ...fno.layers import n_params

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "30"))
OUT = os.environ.get("DOCS_MODES_OUT", "/tmp/docs_modes")

# (label, loader, grid, channels, batch, mode step)
CASES = [
    ("Darcy 32", lambda: data.load_darcy(ROOT, 32), 32, 24, 8, 4),
    ("Darcy 64", lambda: data.load_darcy(ROOT, 64), 64, 24, 8, 4),
    ("Burgers", lambda: data.load_burgers(ROOT), 16, 24, 16, 2),
]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = {jax.config.jax_default_matmul_precision} | epochs = {EPOCHS}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    rows = []

    for label, loader, grid, channels, batch, step in CASES:
        try:
            xtr, ytr, xte, yte = loader()
        except Exception as exc:
            print(f"\n{label}: SKIPPED ({type(exc).__name__}: {str(exc)[:60]})", flush=True)
            continue
        # the band needs 2*m rows, so m is capped by the SHORTER spatial axis
        m_max = min(xtr.shape[-2], xtr.shape[-1]) // 2
        modes = list(range(step, m_max + 1, step))
        print(f"\n### {label}  grid {xtr.shape[-2]}x{xtr.shape[-1]}  channels={channels} "
              f"batch={batch}  n_modes {modes}", flush=True)
        print(f"  {'m/corner':>9}{'rel L2':>12}{'infer us/sample':>18}{'params':>12}{'train s':>10}",
              flush=True)

        for m in modes:
            t0 = time.perf_counter()
            P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=m, channels=channels,
                                       n_layers=4, steps=EPOCHS, batch=batch, verbose=False,
                                       save=os.path.join(OUT, f"{label.replace(' ','')}_m{m}.npz"))
            us = train.infer_us_per_sample(P, xte, m)
            rows.append({"case": label, "grid": int(xtr.shape[-2]), "n_modes": m,
                         "channels": channels, "batch": batch, "epochs": EPOCHS,
                         "rel_l2": rel, "infer_us": us, "params": n_params(P),
                         "train_s": secs, "wall_s": time.perf_counter() - t0})
            print(f"  {m:>8}{rel:>12.4e}{us:>18.2f}{n_params(P):>12,}{secs:>10.1f}", flush=True)

        with open(os.path.join(OUT, "modes.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUT, "modes.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    bundle = "/tmp/docs_modes_all.tar.gz"
    subprocess.run(["tar", "czf", bundle, "-C", os.path.dirname(OUT), os.path.basename(OUT)],
                   check=False)
    npz = [f for f in os.listdir(OUT) if f.endswith(".npz")]
    print(f"\n!! FETCH BEFORE DELETING THE VM: {bundle}  ({len(npz)} trained models + csv)",
          flush=True)
    print("MODES_DONE", flush=True)


if __name__ == "__main__":
    main()
