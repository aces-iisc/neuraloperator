"""Per-operation device time from xplane traces, rolled up into hardware buckets.

parse_traces.py gives the TOTAL device time of a run; this gives the split across MXU, VPU,
copies and DMA, which is what the breakdown tables report.  REPS must match the bench that
wrote the traces, otherwise every number is scaled wrong.

  parse_ops.py <trace-root> <REPS> [--ops]     --ops lists individual operations too
"""
import collections, glob, os, sys

from tensorflow.tsl.profiler.protobuf import xplane_pb2

from .bench_jax import _hlo_bucket


def _stat_names(plane):
    return {m.id: m.name for m in plane.stat_metadata.values()}


def op_times(trace_dir, reps):
    """[(op_name, category, us_per_run)] for one trace directory, device time only."""
    traces = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    if not traces:
        return []
    space = xplane_pb2.XSpace()
    space.ParseFromString(open(traces[-1], "rb").read())
    plane = next((p for p in space.planes if p.name == "/device:TPU:0"), None)
    if plane is None:
        return []

    stat_name = _stat_names(plane)

    def category_of(stats):
        for stat in stats:
            if stat_name.get(stat.metadata_id) == "hlo_category":
                return stat.str_value or ""
        return None

    # hlo_category is attached to the event METADATA, not to each event occurrence
    meta = {m.id: (m.name, category_of(m.stats)) for m in plane.event_metadata.values()}
    out = []
    for line in plane.lines:
        if line.name != "XLA Ops":
            continue
        for event in line.events:
            name, category = meta.get(event.metadata_id, ("?", None))
            if category is None:
                category = category_of(event.stats) or ""
            out.append((name, category, event.duration_ps / 1e6 / reps))
    return out


def bucket_times(trace_dir, reps):
    """{bucket: us_per_run} plus a 'total' key."""
    totals = collections.Counter()
    for _, category, us in op_times(trace_dir, reps):
        totals[_hlo_bucket(category)] += us
    totals["total"] = sum(v for k, v in totals.items() if k != "total")
    return dict(totals)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    root, reps = sys.argv[1], int(sys.argv[2])
    show_ops = "--ops" in sys.argv
    for trace_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(trace_dir):
            continue
        buckets = bucket_times(trace_dir, reps)
        if not buckets.get("total"):
            continue
        name = os.path.basename(trace_dir)
        parts = "  ".join(f"{k}={v:.2f}" for k, v in sorted(buckets.items()) if k != "total")
        print(f"{name:<44}{buckets['total']:>10.2f} us   {parts}")
        if show_ops:
            for op, category, us in sorted(op_times(trace_dir, reps), key=lambda t: -t[2])[:12]:
                print(f"      {us:>9.3f}  {_hlo_bucket(category):<18}{op[:60]}")


if __name__ == "__main__":
    main()
