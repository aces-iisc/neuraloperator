"""Four architecture probes: lane width, channel alignment, add_grid hoisting, HBM traffic."""
import csv, os, time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno import layers, model

LOGDIR = os.environ.get("DOCS_PROF_DIR", "/tmp/docs_prof_arch")
CSV = os.environ.get("DOCS_ARCH_CSV", "/tmp/docs_arch.csv")
FIELDS = ["probe", "case", "variant", "wall_us", "per_elem_ns", "hbm_mb", "status"]
REPS = 20
CASES = [("Darcy-32", 24, 16, 32, 8), ("NS-128", 64, 64, 128, 8)]


def save(rows):
    with open(CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)


def timed(fn, *a):
    j = jax.jit(fn)
    jax.block_until_ready(j(*a))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); jax.block_until_ready(j(*a)); ts.append(time.perf_counter()-t0)
    return sorted(ts)[len(ts)//2] * 1e6


def hbm(fn, *a):
    try:
        c = jax.jit(fn).lower(*a).compile().cost_analysis()
        if isinstance(c, list): c = c[0]
        return float(c.get("bytes accessed", 0.0)) / 1e6
    except Exception:
        return float("nan")


def probe_lanes(rows):
    """Does the minor axis width change per-element elementwise cost? Tests the lane claim."""
    print("\n=== 1. lane width: same element count, different minor-axis width ===", flush=True)
    print(f"  {'shape':<22}{'minor W':>9}{'wall us':>10}{'ns/elem':>10}", flush=True)
    total = 8 * 64 * 128 * 128
    for W in (16, 32, 64, 128, 256):
        rest = total // W
        x = jnp.asarray(np.random.default_rng(0).standard_normal((rest, W)).astype(np.float32))
        fn = lambda z: jax.nn.gelu(z) * 1.0001 + 0.5
        t = timed(fn, x)
        ns = t * 1e3 / total
        print(f"  {f'({rest}, {W})':<22}{W:>9}{t:>10.1f}{ns:>10.4f}", flush=True)
        rows.append({"probe": "lanes", "case": f"W={W}", "variant": "gelu+mul+add",
                     "wall_us": t, "per_elem_ns": ns, "hbm_mb": float("nan"), "status": "ok"})
        save(rows)


def probe_channels(rows):
    """Padding channels to a multiple of 8. Timing only -- accuracy would need retraining."""
    print("\n=== 2. channel alignment (timing only; C=32 is a different model) ===", flush=True)
    print(f"  {'case':<10}{'channels':>9}{'wall us':>10}{'HBM MB':>10}", flush=True)
    for label, C, nm, g, B in CASES:
        for Cv in sorted({C, (C + 7) // 8 * 8, 32 if C == 24 else C}):
            P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, channels=Cv, n_modes=nm, n_layers=4)
            x = jnp.asarray(np.random.default_rng(0).standard_normal((B, 1, g, g)).astype(np.float32))
            fn = lambda z: model.forward(P, z, nm)
            t, h = timed(fn, x), hbm(fn, x)
            tag = "current" if Cv == C else f"padded {C}->{Cv}"
            print(f"  {label:<10}{Cv:>9}{t:>10.1f}{h:>10.2f}   {tag}", flush=True)
            rows.append({"probe": "channels", "case": label, "variant": f"C={Cv}",
                         "wall_us": t, "per_elem_ns": float("nan"), "hbm_mb": h, "status": "ok"})
            save(rows)


def probe_addgrid(rows):
    """add_grid rebuilds two linspaces, a stack, a broadcast and a concat every call."""
    print("\n=== 3. add_grid: rebuilt per call vs passed in ===", flush=True)
    print(f"  {'case':<10}{'variant':<16}{'wall us':>10}{'HBM MB':>10}", flush=True)
    for label, C, nm, g, B in CASES:
        P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, channels=C, n_modes=nm, n_layers=4)
        x = jnp.asarray(np.random.default_rng(0).standard_normal((B, 1, g, g)).astype(np.float32))
        grid = np.asarray(layers.add_grid(x))[:, 1:]           # the 2 coordinate channels
        gj = jnp.asarray(grid)

        def with_build(z):
            return model.forward(P, z, nm)

        def with_given(z, gg):
            zz = jnp.concatenate([z, gg], axis=1)
            zz = layers.channel_mlp(P["lift"], zz)
            for block in P["blocks"]:
                from ...fno.spectral import spectral_conv
                zz = layers.gelu(spectral_conv(block["filter"], zz, nm)
                                 + layers.conv1x1(block["skip"], zz))
            return layers.channel_mlp(P["proj"], zz)

        for nm_, fn, args in (("rebuilt", with_build, (x,)), ("passed in", with_given, (x, gj))):
            t, h = timed(fn, *args), hbm(fn, *args)
            print(f"  {label:<10}{nm_:<16}{t:>10.1f}{h:>10.2f}", flush=True)
            rows.append({"probe": "addgrid", "case": label, "variant": nm_, "wall_us": t,
                         "per_elem_ns": float("nan"), "hbm_mb": h, "status": "ok"})
            save(rows)


def probe_hbm(rows):
    """How much HBM the whole forward moves, against the minimum it could move."""
    print("\n=== 4. HBM traffic: is there room for an on-chip fused layer? ===", flush=True)
    print(f"  {'case':<10}{'HBM MB':>10}{'in+out MB':>11}{'ratio':>8}{'working set MB':>16}", flush=True)
    for label, C, nm, g, B in CASES:
        P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, channels=C, n_modes=nm, n_layers=4)
        x = jnp.asarray(np.random.default_rng(0).standard_normal((B, 1, g, g)).astype(np.float32))
        fn = lambda z: model.forward(P, z, nm)
        h = hbm(fn, x)
        io = 2 * B * 1 * g * g * 4 / 1e6
        ws = B * C * g * g * 4 / 1e6
        print(f"  {label:<10}{h:>10.2f}{io:>11.3f}{h/io:>8.1f}x{ws:>16.2f}", flush=True)
        rows.append({"probe": "hbm", "case": label, "variant": "full forward", "wall_us": float("nan"),
                     "per_elem_ns": float("nan"), "hbm_mb": h, "status": f"io={io:.3f} ws={ws:.2f}"})
        save(rows)


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print("precision =", jax.config.jax_default_matmul_precision, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    rows = []
    probe_lanes(rows); probe_channels(rows); probe_addgrid(rows); probe_hbm(rows)
    save(rows)
    print("\nARCH_DONE", flush=True)


if __name__ == "__main__":
    main()
