"""
Prints what an xprof trace spent its device time on, by op category and by individual op.

    python tpu/experimental/bench/show_trace.py <trace-dir> [more dirs ...]
    python tpu/experimental/bench/show_trace.py tpu/traces/docs_prof2/fft1d_N=256_reim_f32
    python tpu/experimental/bench/show_trace.py --ops tpu/traces/docs_prof2/fft1d_N=256_reduced_f32

A trace directory is the one holding plugins/profile/<run>/*.xplane.pb. Times are divided by REPS
(the number of iterations captured inside the trace) to give microseconds per run.

Needs TensorFlow only to parse the protobuf:  uv pip install --python .venv/bin/python tensorflow-cpu
"""
import argparse, collections, glob, os, sys

REPS = int(os.environ.get("TRACE_REPS", "20"))


def read(trace_dir):
    xps = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    if not xps:
        raise SystemExit(f"no *.xplane.pb under {trace_dir}")
    from tensorflow.tsl.profiler.protobuf import xplane_pb2
    sp = xplane_pb2.XSpace()
    sp.ParseFromString(open(xps[-1], "rb").read())
    tpu = next((p for p in sp.planes if p.name == "/device:TPU:0"), None)
    if tpu is None:
        raise SystemExit(f"no /device:TPU:0 plane in {xps[-1]}")
    sid = {k: v.name for k, v in tpu.stat_metadata.items()}
    val = lambda st: getattr(st, st.WhichOneof("value")) if st.WhichOneof("value") else 0
    meta = {}
    for mid, em in tpu.event_metadata.items():
        d = {sid.get(s.metadata_id, ""): val(s) for s in em.stats}
        meta[mid] = (d.get("hlo_category", "") or "?", em.name.split(" = ")[0].lstrip("%"),
                     d.get("flops", 0) or 0, d.get("bytes_accessed", 0) or 0)
    by_cat, by_op = collections.Counter(), {}
    for ln in tpu.lines:
        if ln.name != "XLA Ops":
            continue
        for ev in ln.events:
            cat, name, fl, by = meta.get(ev.metadata_id, ("?", "?", 0, 0))
            us = ev.duration_ps / 1e6 / REPS
            by_cat[cat] += us
            e = by_op.setdefault((cat, name), [0.0, 0, fl, by])
            e[0] += us; e[1] += 1
    return by_cat, by_op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ops", action="store_true", help="also list the individual ops")
    a = ap.parse_args()
    for d in a.dirs:
        by_cat, by_op = read(d)
        total = sum(by_cat.values())
        print(f"\n=== {d}\n    {total:.2f} us/run over {REPS} iterations")
        print(f"    {'category':<24}{'us':>9}{'share':>8}")
        for c, v in by_cat.most_common():
            print(f"    {c:<24}{v:>9.2f}{100*v/total:>7.1f}%")
        if a.ops:
            print(f"\n    {'category':<22}{'op':<26}{'us':>9}{'GFLOP':>9}")
            for (c, nm), (us, cnt, fl, by) in sorted(by_op.items(), key=lambda kv: -kv[1][0])[:12]:
                print(f"    {c:<22}{nm[:25]:<26}{us:>9.2f}{fl/1e9:>9.3f}")


if __name__ == "__main__":
    main()
