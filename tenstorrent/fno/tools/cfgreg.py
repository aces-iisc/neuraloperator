#!/usr/bin/env python3
"""Where a named Tensix configuration field lives: which cfg register, which bits.

The field names in the ISA documentation and in the simulator are not register numbers, and the
firmware has to write register numbers.  This reads the simulator's own register description so
the mapping is never transcribed by hand.

    tools/cfgreg.py THCON_SEC1_REG2_Disable_zero_compress
"""
import json
import pathlib
import sys

DEFAULT = pathlib.Path(__file__).resolve().parents[2] / "ttsim/data/bh/tensix_regs.json"


def fields(path=DEFAULT):
    data = json.loads(pathlib.Path(path).read_text())
    for space in ("cfg", "thread_cfg"):
        for reg, entries in data[space].items():
            for entry in entries:
                yield space, int(reg), entry["name"], entry["shift"], entry["size"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    needles = [arg.lower() for arg in sys.argv[1:]]
    for space, reg, name, shift, size in fields():
        if any(needle in name.lower() for needle in needles):
            print(f"{space:10s} cfg({reg:3d})  bits {shift:2d}..{shift + size - 1:2d}  {name}")


if __name__ == "__main__":
    main()
