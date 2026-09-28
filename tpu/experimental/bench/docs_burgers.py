"""Burgers n_modes sweep at a longer training budget, every value the grid can express."""
import csv, json, os, subprocess, time
import jax
from ...fno import data, train
from ...fno.layers import n_params

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "60"))
OUT = os.environ.get("DOCS_BURGERS_OUT", "/tmp/docs_burgers")
MODES = [int(x) for x in os.environ.get("DOCS_MODES", "2,3,4,5,6,7,8").split(",")]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = {jax.config.jax_default_matmul_precision} | epochs = {EPOCHS}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    xtr, ytr, xte, yte = data.load_burgers(ROOT)
    print(f"train{xtr.shape} test{xte.shape}  n_modes {MODES}", flush=True)
    print(f"  {'n_modes':>8}{'rel L2':>12}{'infer us/sample':>18}{'params':>11}{'train s':>10}", flush=True)
    rows = []
    for m in MODES:
        P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=m, channels=24, n_layers=4,
                                   steps=EPOCHS, batch=16, verbose=False,
                                   save=os.path.join(OUT, f"Burgers_e{EPOCHS}_m{m}.npz"))
        us = train.infer_us_per_sample(P, xte, m)
        rows.append({"case": "Burgers", "grid": 16, "n_modes": m, "channels": 24, "batch": 16,
                     "epochs": EPOCHS, "rel_l2": rel, "infer_us": us,
                     "params": n_params(P), "train_s": secs})
        print(f"  {m:>8}{rel:>12.4e}{us:>18.2f}{n_params(P):>11,}{secs:>10.1f}", flush=True)
        with open(os.path.join(OUT, "burgers_modes.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT, "burgers_modes.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    bundle = "/tmp/docs_burgers_all.tar.gz"
    subprocess.run(["tar","czf",bundle,"-C",os.path.dirname(OUT),os.path.basename(OUT)], check=False)
    print(f"\n!! FETCH BEFORE DELETING THE VM: {bundle}", flush=True)
    print("BURGERS_DONE", flush=True)


if __name__ == "__main__":
    main()
