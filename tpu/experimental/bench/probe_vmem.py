"""How much on-chip memory this device actually has, read off the compiler rather than a datasheet.

The batch rule divides the on-chip budget by the working set of one sample, so that budget is a
constant the planner cannot afford to guess at -- and it does not follow the generation.  XLA
knows the real figure: ask it to keep an array resident that is too large and the allocation
failure names the limit.  This grows a request until that failure appears and reports what it said.

Run it on the device being characterised, then put the number in `arch.py`.
"""
import re

import jax
import jax.numpy as jnp

# bytes XLA reports in a scratch-allocation failure, in either of the two phrasings it uses
LIMITS = [re.compile(r"([\d,]+)\s*bytes.{0,40}?(?:available|limit|capacity)", re.I),
          re.compile(r"(?:limit|capacity|available)[^\d]{0,40}([\d,]+)\s*bytes", re.I)]


def _reported_limit(message):
    for pattern in LIMITS:
        found = pattern.search(message)
        if found:
            return int(found.group(1).replace(",", ""))
    return None


def probe(start_mib=8, stop_mib=1024):
    """Grow a resident working set until the compiler refuses, and report what it refused at."""
    print(f"{jax.devices()[0].device_kind}  ({len(jax.devices())} device(s))\n", flush=True)
    last_ok = None
    size = start_mib
    while size <= stop_mib:
        elements = size * 2 ** 20 // 4
        try:
            # three live buffers, matching what the layer keeps alive, forced to stay resident
            # by making every one of them feed the result
            def resident(seed, n=elements):
                a = jnp.full((n,), seed, jnp.float32)
                b = a * 2.0
                c = b + a
                return (a.sum() + b.sum() + c.sum())
            jax.block_until_ready(jax.jit(resident).lower(1.0).compile()(1.0))
            last_ok = size
            print(f"  {size:>6} MiB  resident", flush=True)
        except Exception as exc:
            limit = _reported_limit(str(exc))
            print(f"  {size:>6} MiB  refused: {type(exc).__name__}", flush=True)
            if limit:
                print(f"\n  compiler reports a limit of {limit:,} bytes "
                      f"({limit / 2 ** 20:.1f} MiB)", flush=True)
            else:
                print(f"\n  no limit in the message; last size that fit was {last_ok} MiB")
                print(f"  message: {str(exc)[:400]}", flush=True)
            return limit, last_ok
        size *= 2
    print(f"\n  never refused up to {stop_mib} MiB", flush=True)
    return None, last_ok


if __name__ == "__main__":
    probe()
