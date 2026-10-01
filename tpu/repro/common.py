"""Shared harness for the reproduction benches.

One module per table lives beside this one and imports from here, so a measurement is written
once and reused rather than copied.  Everything reports H1 relative error; the FNO objective is
H1, so L2 is not carried anywhere in this package.

Each table module exposes:
    TABLE   the \\label{tab:...} it fills
    FIELDS  its CSV columns
    run()   writes its CSV and returns the rows

`run_all.py` discovers them, runs each exactly once, and collects the outputs.
"""
import csv, json, os, time

import jax
import jax.numpy as jnp
import numpy as np

from ..fft.core import BF, F
from ..fno import data, layers, model

ROOT = os.path.expanduser(os.environ.get("DOCS_DATA", "~/data"))
OUT = os.path.expanduser(os.environ.get("REPRO_OUT", "~/repro_out"))
WEIGHTS = os.path.expanduser(os.environ.get("REPRO_WEIGHTS", "~/repro_out/weights"))
TRACES = os.path.expanduser(os.environ.get("REPRO_TRACES", "~/repro_out/traces"))
REPS = int(os.environ.get("REPRO_REPS", "15"))

# (label, loader, channels, n_modes, batch for timing).  The four shipped datasets.
CASES = [
    ("Burgers",  lambda: data.load_burgers(ROOT),                                24, 16, 128),
    ("Darcy-32", lambda: data.load_darcy(ROOT, 32),                              24, 16, 128),
    ("Darcy-64", lambda: data.load_darcy(ROOT, 64),                              24, 16, 128),
    ("NS-128",   lambda: data.load_ns(ROOT, 128, n_train=1000, n_test=200),      64, 64, 8),
]
CASE_BY_NAME = {c[0]: c for c in CASES}


def ensure_dirs():
    for d in (OUT, WEIGHTS, TRACES):
        os.makedirs(d, exist_ok=True)


def load_case(label):
    """(x_train, y_train, x_test, y_test) for one dataset, or None if its files are absent."""
    try:
        return CASE_BY_NAME[label][1]()
    except Exception as exc:
        print(f"    {label}: data unavailable ({type(exc).__name__})", flush=True)
        return None


# ── metrics ─────────────────────────────────────────────────────────────────────────
def h1(pred, y):
    """Relative H1 error, the objective the FNO is actually trained against."""
    return float(layers.h1_rel(jnp.asarray(pred), jnp.asarray(y)))


def h1_batched(forward, x, y, batch):
    """H1 over a test split too large to hold in one forward."""
    num = den = 0.0
    for start in range(0, x.shape[0], batch):
        xs, ys = jnp.asarray(x[start:start + batch]), jnp.asarray(y[start:start + batch])
        pred = forward(xs)
        num += float(jnp.sum((pred - ys) ** 2))
        den += float(jnp.sum(ys ** 2))
    return float(np.sqrt(num / den))


# ── timing ──────────────────────────────────────────────────────────────────────────
def timed(fn, args, tag, reps=None):
    """Median wall clock in us, with an xprof trace written under TRACES/tag.

    Wall clock has a dispatch floor of roughly 180 us on this machine, so anything small is
    read from the trace instead; `device_us` recovers it from the same capture.
    """
    reps = reps or REPS
    ensure_dirs()
    jax.block_until_ready(fn(*args))
    samples = []
    with jax.profiler.trace(os.path.join(TRACES, tag)):
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            samples.append(time.perf_counter() - t0)
    return sorted(samples)[reps // 2] * 1e6


# ── trained weights, produced once and reused ───────────────────────────────────────
def weight_file(label, tag="ours"):
    return os.path.join(WEIGHTS, f"{label}_{tag}.npz")


def have_weights(label, tag="ours"):
    return os.path.exists(weight_file(label, tag))


def load_weights(label, tag="ours"):
    _, _, channels, n_modes, _ = CASE_BY_NAME[label]
    return model.load_params(weight_file(label, tag), channels=channels, n_modes=n_modes)


# ── output ──────────────────────────────────────────────────────────────────────────
def write(rows, name, fields):
    ensure_dirs()
    path = os.path.join(OUT, f"{name}.csv")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}  ({len(rows)} rows)", flush=True)
    return path


def chain(rows, key="us_per_sample", order=None):
    """Annotate an ordered ladder with each step's gain over the PREVIOUS step.

    `gain_prev` is what the step itself bought; `gain_base` is the running product from the
    first row.  Reporting only the second hides which step did the work.
    """
    order = order or list(range(len(rows)))
    first = prev = None
    for i in order:
        value = rows[i].get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            rows[i]["gain_prev"] = rows[i]["gain_base"] = float("nan")
            continue
        if prev is None:
            first = prev = value
            rows[i]["gain_prev"] = rows[i]["gain_base"] = 1.0
        else:
            rows[i]["gain_prev"] = prev / value
            rows[i]["gain_base"] = first / value
            prev = value
    return rows
