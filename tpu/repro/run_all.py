"""Run every reproduction bench once, in registry order.

  python -m tpu.repro.run_all                  everything
  python -m tpu.repro.run_all ladder results   only entries whose name or table matches
  python -m tpu.repro.run_all --list           what produces what, run nothing

registry.py holds the table -> bench mapping, so each measurement runs exactly once even when
it fills several tables.  Weights are trained first and shared; nothing retrains.
"""
import importlib, inspect, os, sys, time

from .common import Paths
from .registry import CSV_ENV, ENTRIES, STATIC


def _entry_point(mod):
    """The module's no-argument entry point.

    `main` is preferred: several benches also define a `run(tag, fn, x)` timing helper, and
    picking that by name calls the wrong thing with no arguments.
    """
    for name in ("main", "run"):
        fn = getattr(mod, name, None)
        if fn is None:
            continue
        required = [p for p in inspect.signature(fn).parameters.values()
                    if p.default is p.empty and p.kind in
                    (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if not required:
            return fn
    return None


def _has_tensorflow():
    try:
        import tensorflow                       # noqa: F401
        return True
    except Exception:
        return False


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
    Paths.ensure()
    # "slow" entries cost hours and fill a figure rather than a table; ask for them by name
    chosen = [e for e in ENTRIES
              if (any(a in e[0] or any(a in t for t in e[1]) for a in args) if args
                  else "slow" not in e[2])]
    print(f"reproduction: {len(chosen)} entr(y/ies) -> {Paths.out}\n", flush=True)
    summary = []
    for module, tables, needs, _ in chosen:
        name = module.split(".")[-1]
        if "tensorflow" in needs and not _has_tensorflow():
            print(f"  -- {name}: needs TensorFlow to read the traces; run it where that is "
                  f"installed", flush=True)
            summary.append((name, ", ".join(tables), "deferred: no tensorflow", 0.0))
            continue
        t0 = time.perf_counter()
        try:
            # route this bench's output into REPRO_OUT instead of wherever it defaults to
            if name in CSV_ENV:
                os.environ[CSV_ENV[name]] = os.path.join(Paths.out, f"{name}.csv")
                os.environ.setdefault("DOCS_PROF_DIR", Paths.traces)
            mod = importlib.import_module(module, __package__)
            entry = _entry_point(mod)
            if entry is None:
                raise RuntimeError("no zero-argument main() or run()")
            entry()
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
