"""
Trains the same FNO two ways on each dataset and compares them: our real/imag spectral conv
against a baseline built from JAX's own primitives (jnp.fft.rfft2, a complex64 einsum for the
channel mix, jnp.fft.irfft2).

Both share the weight tree, the initialisation seed, the optimizer and the epoch budget, so the
only difference is how the spectral convolution is computed. Weights are saved for every run.

    PYTHONPATH=~/nop DOCS_EPOCHS=60 ~/nopenv/bin/python -m tpu.experimental.bench.docs_baseline
"""
import csv, json, os, subprocess, time
import jax

from ...fno import data, train
from ...fno.layers import n_params
from ...fft.core import BF
from ...fno.model import forward as ours
from ..fno.reference import forward as jnp_baseline

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
EPOCHS = int(os.environ.get("DOCS_EPOCHS", "60"))
OUT = os.environ.get("DOCS_BASE_OUT", "/tmp/docs_baseline")

# (label, loader, n_modes, channels, batch, preset)   n_modes is the TOTAL band
CASES = [
    ("Burgers", lambda: data.load_burgers(ROOT), 16, 24, 16, "FNO_Small2d"),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32), 16, 24, 8, "FNO_Small2d"),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64), 16, 24, 8, "FNO_Small2d"),
    ("NS-128", lambda: data.load_ns(ROOT, 128, n_train=10000, n_test=1000), 64, 64, 8,
     "FNO_Medium2d"),
]
ENGINES = [("ours", ours),
           ("ours-bf16", lambda P, x, m: ours(P, x, m, BF)),
           ("jnp", jnp_baseline)]

# DOCS_ENGINES=ours-bf16 runs just that one, so a variant can be added without retraining the rest
_want = os.environ.get("DOCS_ENGINES")
if _want:
    keep = {w.strip() for w in _want.split(",")}
    ENGINES = [e for e in ENGINES if e[0] in keep]


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"precision = {jax.config.jax_default_matmul_precision} | epochs = {EPOCHS}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    rows = []

    for label, loader, n_modes, channels, batch, preset in CASES:
        try:
            xtr, ytr, xte, yte = loader()
        except Exception as exc:
            print(f"\n{label}: SKIPPED ({type(exc).__name__}: {str(exc)[:60]})", flush=True)
            continue
        print(f"\n### {label}  {preset}  n_modes={n_modes} channels={channels} batch={batch}  "
              f"train{xtr.shape} test{xte.shape}", flush=True)
        print(f"  {'engine':<8}{'rel L2':>12}{'train s':>10}{'infer us/sample':>18}{'params':>12}",
              flush=True)

        for name, fwd in ENGINES:
            t0 = time.perf_counter()
            P, rel, secs = train.train(xtr, ytr, xte, yte, n_modes=n_modes, channels=channels,
                                       n_layers=4, steps=EPOCHS, batch=batch, fwd=fwd,
                                       verbose=False,
                                       save=os.path.join(OUT, f"{label}_{name}.npz"),
                                       log_csv=os.path.join(OUT, f"{label}_{name}.csv"))
            us = train.infer_us_per_sample(P, xte, n_modes, fwd=fwd)
            rows.append({"case": label, "preset": preset, "engine": name, "n_modes": n_modes,
                         "channels": channels, "batch": batch, "epochs": EPOCHS,
                         "rel_l2": rel, "train_s": secs, "infer_us": us,
                         "params": n_params(P), "wall_s": time.perf_counter() - t0})
            print(f"  {name:<8}{rel:>12.4e}{secs:>10.1f}{us:>18.2f}{n_params(P):>12,}", flush=True)
            with open(os.path.join(OUT, "baseline.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
                w.writeheader(); w.writerows(rows)

    with open(os.path.join(OUT, "baseline.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\n{'case':<10}{'ours':>12}{'ours-bf16':>12}{'jnp':>12}   |{'ours us':>9}"
          f"{'bf16 us':>9}{'jnp us':>10}{'bf16 vs jnp':>13}", flush=True)
    for label, *_ in CASES:
        g = {r['engine']: r for r in rows if r['case'] == label}
        if len(g) < 3:
            continue  # summary needs all three; a filtered run prints none
        print(f"{label:<10}{g['ours']['rel_l2']:>12.3e}{g['ours-bf16']['rel_l2']:>12.3e}"
              f"{g['jnp']['rel_l2']:>12.3e}   |{g['ours']['infer_us']:>9.2f}"
              f"{g['ours-bf16']['infer_us']:>9.2f}{g['jnp']['infer_us']:>10.2f}"
              f"{g['jnp']['infer_us']/g['ours-bf16']['infer_us']:>12.2f}x", flush=True)
    bundle = "/tmp/docs_baseline_all.tar.gz"
    subprocess.run(["tar","czf",bundle,"-C",os.path.dirname(OUT),os.path.basename(OUT)], check=False)
    npz = [f for f in os.listdir(OUT) if f.endswith('.npz')]
    print(f"\n!! FETCH BEFORE DELETING THE VM: {bundle}  ({len(npz)} trained models)", flush=True)
    print("BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
