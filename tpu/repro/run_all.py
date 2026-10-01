"""Run every reproduction bench once, in registry order.

  python -m tpu.repro.run_all                  everything
  python -m tpu.repro.run_all ladder results   only entries whose name or table matches
  python -m tpu.repro.run_all --list           what produces what, run nothing

registry.py holds the table -> bench mapping, so each measurement runs exactly once even when
it fills several tables.  Weights are trained first and shared; nothing retrains.
"""
import importlib, sys, time

from . import common
from .registry import ENTRIES, STATIC


def show():
    print(f"{'module':<40}{'needs':<18}tables")
    for module, tables, needs, note in ENTRIES:
        print(f"{module:<40}{','.join(needs) or '-':<18}{', '.join(tables) or '(weights)'}")
        print(f"{'':<40}{'':<18}{note}")
    print("\nnot measured here:")
    for table, why in STATIC.items():
        print(f"  {table:<22}{why}")


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    if "--list" in argv:
        show()
        return
    common.ensure_dirs()
    chosen = [e for e in ENTRIES
              if not args or any(a in e[0] or any(a in t for t in e[1]) for a in args)]
    print(f"reproduction: {len(chosen)} entr(y/ies) -> {common.OUT}\n", flush=True)
    summary = []
    for module, tables, needs, _ in chosen:
        name = module.split(".")[-1]
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(module, __package__)
            (mod.run if hasattr(mod, "run") else mod.main)()
            status = "ok"
        except Exception as exc:
            status = f"{type(exc).__name__}: {str(exc)[:50]}"
            print(f"  !! {name}: {exc}", flush=True)
        summary.append((name, ", ".join(tables) or "(weights)", status,
                        time.perf_counter() - t0))
    print("\n=== summary ===", flush=True)
    for name, tables, status, secs in summary:
        print(f"  {name:<24}{tables:<46}{status:<28}{secs:>7.0f} s", flush=True)
    print("REPRO_ALL_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv)
