"""Does hoisting pay? It depends on whether the weights reach the kernel as constants."""
import time

import jax, jax.numpy as jnp, numpy as np

from ...fft.core import F
from ...fno import layers, model

REPS = 20
CASES = [("Darcy-32", 24, 16, (32, 32), 256), ("NS-128", 64, 64, (128, 128), 8)]


def timed(fn, *args):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[REPS // 2] * 1e6


def main():
    print("jax", jax.__version__, "|", jax.devices(), flush=True)
    print(f"{'case':<10}{'weights reach kernel as':<26}{'w_block':<10}{'us/call':>10}{'us/sample':>11}",
          flush=True)
    for label, C, nm, (H, W), B in CASES:
        P = layers.init_fno(jax.random.PRNGKey(0), 1, 1, C, nm, 4)
        Pb = layers.prebuild_blocks(P, F)
        x = jax.random.normal(jax.random.PRNGKey(1), (B, 1, H, W), F)

        variants = [
            ("captured constant", "rebuilt", jax.jit(lambda z, _p=P: model.forward(_p, z, nm)), (x,)),
            ("captured constant", "prebuilt", jax.jit(lambda z, _p=Pb: model.forward(_p, z, nm)), (x,)),
            ("runtime argument", "rebuilt", jax.jit(lambda p, z: model.forward(p, z, nm)), (P, x)),
            ("runtime argument", "prebuilt", jax.jit(lambda p, z: model.forward(p, z, nm)), (Pb, x)),
        ]
        base = None
        for how, blk, fn, args in variants:
            try:
                us = timed(fn, *args)
                if base is None:
                    base = us
                print(f"{label:<10}{how:<26}{blk:<10}{us:>10.1f}{us / B:>11.2f}", flush=True)
            except Exception as exc:
                print(f"{label:<10}{how:<26}{blk:<10}  FAILED {str(exc)[:60]}", flush=True)
        print(flush=True)
    print("ARGCONST_DONE", flush=True)


if __name__ == "__main__":
    main()
