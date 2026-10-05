"""A dependency-free reader for the xplane traces, enough for the per-operation tables.

The usual reader is `tensorflow.tsl.profiler.protobuf.xplane_pb2`, which means installing
TensorFlow just to decode a few fields.  The protobuf wire format carries field numbers and wire
types inline, so the handful of fields the breakdown tables need can be decoded directly.

Field numbers are not taken on trust.  `verify` locates them by content -- the device plane is the
one whose name is "/device:TPU:0", the operation line is the one named "XLA Ops" -- and reports
what it found, so a layout that does not match is visible rather than silently mis-decoded.
"""
import collections
import struct

# xplane.proto, confirmed against the traces by `verify` below.
XSPACE_PLANES = 1
PLANE_NAME, PLANE_LINES, PLANE_EVENT_META, PLANE_STAT_META = 2, 3, 4, 5
LINE_NAME, LINE_EVENTS = 2, 4
EVENT_METADATA_ID, EVENT_DURATION_PS, EVENT_STATS = 1, 3, 4
STAT_METADATA_ID, STAT_STR_VALUE = 1, 5
META_ID, META_NAME, META_STATS = 1, 2, 5


def _varint(buf, i):
    value = shift = 0
    while True:
        byte = buf[i]
        i += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, i
        shift += 7


def fields(buf):
    """{field_number: [payload, ...]} for one message.

    Length-delimited payloads come back as memoryview slices, varints as ints.  Repeated fields
    keep every occurrence, which is what `lines`, `events` and the map entries need.
    """
    out = collections.defaultdict(list)
    i, end = 0, len(buf)
    while i < end:
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
            out[number].append(value)
        elif wire == 1:
            out[number].append(struct.unpack_from("<Q", buf, i)[0])
            i += 8
        elif wire == 2:
            length, i = _varint(buf, i)
            out[number].append(buf[i:i + length])
            i += length
        elif wire == 5:
            out[number].append(struct.unpack_from("<I", buf, i)[0])
            i += 4
        else:                                    # groups: not used by xplane
            raise ValueError(f"unsupported wire type {wire} for field {number}")
    return out


def _text(payload):
    return bytes(payload).decode("utf-8", "replace")


def _map_entries(payloads):
    """A proto3 map field decodes as repeated entries of {1: key, 2: value}."""
    for payload in payloads:
        entry = fields(payload)
        if 2 in entry:
            yield entry[2][0]


def planes(path):
    """Every XPlane in the file, as decoded field dicts."""
    with open(path, "rb") as handle:
        space = fields(memoryview(handle.read()))
    return [fields(p) for p in space[XSPACE_PLANES]]


def device_plane(path, name="/device:TPU:0"):
    for plane in planes(path):
        if plane.get(PLANE_NAME) and _text(plane[PLANE_NAME][0]) == name:
            return plane
    return None


def op_times(path, reps, line_name="XLA Ops"):
    """[(op_name, hlo_category, us_per_run)] from one trace file, device time only."""
    plane = device_plane(path)
    if plane is None:
        return []

    stat_names = {}
    for payload in _map_entries(plane.get(PLANE_STAT_META, [])):
        meta = fields(payload)
        stat_names[meta[META_ID][0]] = _text(meta[META_NAME][0])

    def category_of(stat_payloads):
        for payload in stat_payloads:
            stat = fields(payload)
            if stat_names.get(stat[STAT_METADATA_ID][0]) == "hlo_category":
                return _text(stat[STAT_STR_VALUE][0]) if STAT_STR_VALUE in stat else ""
        return None

    # hlo_category hangs off the event metadata, not off each occurrence
    event_meta = {}
    for payload in _map_entries(plane.get(PLANE_EVENT_META, [])):
        meta = fields(payload)
        name = _text(meta[META_NAME][0]) if META_NAME in meta else "?"
        event_meta[meta[META_ID][0]] = (name, category_of(meta.get(META_STATS, [])))

    out = []
    for line_payload in plane.get(PLANE_LINES, []):
        line = fields(line_payload)
        if not line.get(LINE_NAME) or _text(line[LINE_NAME][0]) != line_name:
            continue
        for event_payload in line.get(LINE_EVENTS, []):
            event = fields(event_payload)
            name, category = event_meta.get(event[EVENT_METADATA_ID][0], ("?", None))
            if category is None:
                category = category_of(event.get(EVENT_STATS, [])) or ""
            duration = event[EVENT_DURATION_PS][0] if EVENT_DURATION_PS in event else 0
            out.append((name, category, duration / 1e6 / reps))
    return out


def verify(path):
    """What the decoder found in one file, for checking the field numbers hold.

    Returns a dict naming the planes, the lines on the device plane, how many events and
    categories were read, and the total device time, so the numbers can be read against the
    wall time the bench recorded for the same run.
    """
    found = {"planes": [_text(p[PLANE_NAME][0]) for p in planes(path) if p.get(PLANE_NAME)]}
    plane = device_plane(path)
    if plane is None:
        return found
    found["device_lines"] = [_text(fields(p)[LINE_NAME][0]) for p in plane.get(PLANE_LINES, [])
                             if fields(p).get(LINE_NAME)]
    ops = op_times(path, 1)
    found["n_events"] = len(ops)
    found["categories"] = sorted({c for _, c, _ in ops if c})
    found["total_us_one_rep"] = sum(us for _, _, us in ops)
    return found


def _hlo_bucket(cat):
    """Roll XLA's REAL `hlo_category` up into MXU/VPU/copy/DMA/custom (the ONE bucket list).
    e.g. 'convolution fusion'->MXU (our einsum-DFT), 'loop/custom fusion'->VPU,
    'slice'/'data formatting'/'reverse'->copy, 'copy-done'->DMA.  Category strings come from the
    `hlo_category` XStat on /device:TPU:0 of the *.xplane.pb (xprof — https://openxla.org/xprof;
    proto tensorflow.tsl.profiler.protobuf.xplane_pb2); the labels are XLA's, the roll-up is ours."""
    c = cat.lower()
    if not c:
        return "other"
    if "convolution" in c or "dot" in c or "gemm" in c:
        return "MXU matmul"
    if c in ("copy-start", "copy-done") or any(k in c for k in
                                               ("all-reduce", "infeed", "outfeed", "dma")):
        return "DMA / HBM"
    if "custom-call" in c:
        return "custom-call (fft)"
    if "fusion" in c:  # loop/custom/input/output fusion -> VPU element-wise (incl. gather_fusion)
        return "VPU / fusion"
    if any(k in c for k in ("slice", "data formatting", "reverse", "transpose", "copy",
                            "broadcast", "gather", "concatenate", "pad", "reshape", "bitcast")):
        return "reshape / copy"
    return c  # keep XLA's own label for anything unmapped
