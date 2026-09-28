"""Compares activations by test rel-L2 on CPU; no TPU needed since only accuracy is measured."""
import argparse, csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fno import data, train

ACTS = {
    "gelu": jax.nn.gelu,
    "relu": jax.nn.relu,
    "leaky_relu": lambda x: jax.nn.leaky_relu(x, 0.01),
    "silu": jax.nn.silu,
    "tanh": jnp.tanh,
    "elu": jax.nn.elu,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Darcy-32")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--n-modes", type=int, default=16)
    ap.add_argument("--channels", type=int, default=24)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="tpu/results/activation.csv")
    ap.add_argument("--only", default="", help="comma-separated subset of activations")
    args = ap.parse_args()

    root = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
    res = int(args.case.split("-")[1])
    xtr, ytr, xte, yte = data.load_darcy(root, res)
    xtr, ytr = xtr[:args.n_train], ytr[:args.n_train]
    xte, yte = xte[:args.n_test], yte[:args.n_test]
    print(f"{args.case}: train{tuple(xtr.shape)} test{tuple(xte.shape)} "
          f"n_modes={args.n_modes} channels={args.channels} epochs={args.epochs} "
          f"seeds={args.seeds}", flush=True)
    print(f"\n  {'activation':<12}{'mean rel L2':>13}{'std':>10}{'best':>11}{'train s':>10}",
          flush=True)

    rows = []
    import tpu.fno.layers as layers
    import tpu.fno.model as model
    original = layers.gelu
    want = [w.strip() for w in args.only.split(",") if w.strip()] or list(ACTS)
    for name in want:
        fn = ACTS[name]
        # model.forward and the channel MLP both call the name `gelu`; swap what it points at
        layers.gelu = fn
        model.gelu = fn
        errs, t0 = [], time.perf_counter()
        for seed in range(args.seeds):
            _, rel, _ = train.train(xtr, ytr, xte, yte, n_modes=args.n_modes,
                                    channels=args.channels, n_layers=4, steps=args.epochs,
                                    batch=8, seed=seed, verbose=False)
            errs.append(float(rel))
        secs = time.perf_counter() - t0
        print(f"  {name:<12}{np.mean(errs):>13.4e}{np.std(errs):>10.1e}"
              f"{min(errs):>11.4e}{secs:>10.1f}", flush=True)
        rows.append({"activation": name, "mean_rel_l2": np.mean(errs), "std": np.std(errs),
                     "best": min(errs), "seeds": args.seeds, "epochs": args.epochs,
                     "case": args.case, "train_s": secs})
    layers.gelu, model.gelu = original, original

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    best = min(rows, key=lambda r: r["mean_rel_l2"])
    print(f"\n  best: {best['activation']} at {best['mean_rel_l2']:.4e}", flush=True)
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
