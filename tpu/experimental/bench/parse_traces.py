"""Pull per-run device time out of each xplane trace (the /device:TPU:0 'XLA Ops' line).

REPS must match the bench that wrote the traces, otherwise every number is scaled wrong.
Pass it as argv[4] or set TRACE_REPS; there is no safe default, so this refuses to guess."""
import csv, glob, os, sys
from tensorflow.tsl.profiler.protobuf import xplane_pb2

REPS = None   # set from argv/env below; a wrong value scales every result

def device_us(trace_dir):
    xps = sorted(glob.glob(os.path.join(trace_dir, "**", "*.xplane.pb"), recursive=True))
    if not xps:
        return None
    sp = xplane_pb2.XSpace()
    sp.ParseFromString(open(xps[-1], "rb").read())
    tpu = next((p for p in sp.planes if p.name == "/device:TPU:0"), None)
    if tpu is None:
        return None
    total_ps = 0
    for ln in tpu.lines:
        if ln.name != "XLA Ops":
            continue
        for ev in ln.events:
            total_ps += ev.duration_ps
    if total_ps == 0:
        return None
    return total_ps / 1e6 / REPS          # ps -> us, per run

if len(sys.argv) < 5 and not os.environ.get("TRACE_REPS"):
    raise SystemExit("usage: parse_traces.py <trace-root> <csv-in> <csv-out> <REPS>\n"
                     "REPS must equal the bench's REPS. Refusing to guess.")
root = sys.argv[1]; csv_in = sys.argv[2]; csv_out = sys.argv[3]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ["TRACE_REPS"])
rows = list(csv.DictReader(open(csv_in)))
ok = 0
for r in rows:
    tr = r.get("trace", "").strip()
    d = device_us(os.path.join(root, tr)) if tr else None
    r["device_us"] = f"{d:.2f}" if d else ""
    ok += bool(d)
keys = list(rows[0].keys())
with open(csv_out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
print(f"parsed device time for {ok}/{len(rows)} rows -> {csv_out}")
