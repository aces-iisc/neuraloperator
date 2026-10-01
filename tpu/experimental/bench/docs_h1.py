"""Train our FNO under neuralop's own loss, learning rate and schedule, one run per dataset."""
import csv, os, time

import jax

from ...fno import data, train

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
OUT = os.environ.get("DOCS_H1_OUT", "/tmp/h1w")
CSV = os.environ.get("DOCS_H1_CSV", "/tmp/docs_h1.csv")
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "60"))
FIELDS = ["case", "loss", "lr", "lr_step", "batch", "epochs", "rel_l2", "train_s", "weights"]

# Settings copied from the shipped configs: burgers_config, darcy_config, navier_stokes_config.
CASES = [
    ("Burgers",  lambda: data.load_burgers(ROOT),   24, 16, "h1", 1e-4, 60,  16),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 24, 16, "h1", 5e-3, 60,  8),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 24, 16, "h1", 5e-3, 60,  8),
    ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=10000, n_test=1000),
     64, 64, "h1", 3e-4, 100, 8),
]
WANT = [c.strip() for c in os.environ.get(
    "DOCS_CASES", "Burgers,Darcy-32,Darcy-64,NS-128").split(",")]


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for label, loader, channels, n_modes, loss, lr, lr_step, batch in CASES:
        if label not in WANT:
            continue
        print(f"\n===== {label}: loss={loss} lr={lr} step={lr_step} batch={batch} "
              f"epochs={EPOCHS}", flush=True)
        try:
            xtr, ytr, xte, yte = loader()
        except Exception as exc:
            print(f"  data unavailable: {exc}", flush=True)
            continue
        path = os.path.join(OUT, f"{label}_h1.npz")
        t0 = time.perf_counter()
        _, rel, secs = train.train(
            xtr, ytr, xte, yte, n_modes=n_modes, channels=channels, n_layers=4,
            steps=EPOCHS, batch=batch, lr=lr, weight_decay=1e-4, lr_step=lr_step,
            lr_gamma=0.5, seed=0, loss=loss, log_every=10, name=label, save=path)
        rows.append({"case": label, "loss": loss, "lr": lr, "lr_step": lr_step,
                     "batch": batch, "epochs": EPOCHS, "rel_l2": rel,
                     "train_s": secs, "weights": path})
        print(f"  {label}: rel_l2={rel:.4e}  train={secs:.1f}s  -> {path}", flush=True)
        save(rows)
    save(rows)
    os.system(f"cd /tmp && tar czf /tmp/h1.tar.gz {os.path.basename(OUT)} "
              f"{os.path.basename(CSV)}")
    print("\nH1_DONE", flush=True)


if __name__ == "__main__":
    main()
