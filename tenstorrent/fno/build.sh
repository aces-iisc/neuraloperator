#!/bin/sh
# Builds the tile firmware and the host driver.
#
# The firmware is a flat RISC-V image that BRISC fetches from L1 address 0, so it is compiled
# freestanding, linked at zero, and turned into raw bytes.  The host side is ordinary C++ and
# only needs libdl: it reaches the hardware through libttsim, which it loads at run time.
set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
SFPI=${SFPI:-$ROOT/../toolchain/sfpi}
CROSS=$SFPI/compiler/bin/riscv-tt-elf
OUT=$ROOT/_out
mkdir -p "$OUT"

# -- tile firmware ---------------------------------------------------------------------------
# The architecture string is the one Tenstorrent's own kernels are built with: a Blackhole baby
# RISC-V, which has a single-precision FPU and the bit-manipulation extensions.  Compiling and
# linking in one step keeps the flags in a single place.
FWFLAGS="-Os -g -std=gnu++20 -Wall -Wextra -Wno-unused-parameter \
         -fno-exceptions -fno-rtti -fno-threadsafe-statics \
         -ffreestanding -fno-builtin -fomit-frame-pointer \
         -mcpu=tt-bh -march=rv32imf_zba_zbb_zicsr_zifencei -mabi=ilp32f"

"$CROSS-g++" $FWFLAGS -nostdlib -Wl,-T,"$ROOT/fw/kernel.ld" -Wl,--no-warn-rwx-segments \
    -o "$OUT/kernel.elf" "$ROOT/fw/entry.S" "$ROOT/fw/kernel.cpp"
"$CROSS-objcopy" -O binary "$OUT/kernel.elf" "$OUT/kernel.bin"
"$CROSS-size" "$OUT/kernel.elf"

# -- the three-stage firmware ----------------------------------------------------------------
# The same source three times, once per Tensix thread, each linked at the reset address of the
# TRISC that will run it.  The stage is a compile-time constant, so each image carries only its
# own stage and the shared command loop.
for thread in 0 1 2; do
    case $thread in
        0) base=0x6000 ;;
        1) base=0xA000 ;;
        2) base=0xE000 ;;
    esac
    "$CROSS-g++" $FWFLAGS -DTT_THREAD=$thread -nostdlib \
        -Wl,-T,"$ROOT/fw/trisc.ld" -Wl,--defsym,FW_BASE=$base -Wl,--no-warn-rwx-segments \
        -o "$OUT/stage$thread.elf" "$ROOT/fw/trisc_entry.S" "$ROOT/fw/pipeline.cpp"
    "$CROSS-objcopy" -O binary "$OUT/stage$thread.elf" "$OUT/stage$thread.bin"
    "$CROSS-size" "$OUT/stage$thread.elf"
done

# The host embeds the images rather than reading them at run time, so a built binary is
# self-contained and the firmware can never be out of step with the driver that loads it.
python3 "$ROOT/tools/mkinc.py" "$OUT/kernel.bin" "$OUT/firmware.inc" fw/kernel.cpp
for thread in 0 1 2; do
    python3 "$ROOT/tools/mkinc.py" "$OUT/stage$thread.bin" "$OUT/stage$thread.inc" \
        "fw/pipeline.cpp thread $thread"
done

# -- host ------------------------------------------------------------------------------------
HOSTFLAGS="-O2 -g -std=c++20 -Wall -Wextra -I$OUT"

# The hardware layer: the one matmul the device performs, and what it takes to get operands
# there.  Nothing in it knows what a transform is.
DEVICE="$ROOT/host/device.cpp $ROOT/host/numerics.cpp $ROOT/host/tile_matmul.cpp"

# The transform layer, laid out to mirror torch_fno/fft: shared helpers, then one file per
# variant.  core.cpp holds the row sets, the restricted DFT matrices, the Karatsuba complex
# product and the float64 references.
FFT="$ROOT/host/fft/core.cpp $ROOT/host/fft/rfft1d.cpp $ROOT/host/fft/tricks.cpp \
     $ROOT/host/fft/rfft2d.cpp $ROOT/host/fft/transform.cpp"

g++ $HOSTFLAGS -o "$OUT/check_mvmul" "$ROOT/host/check_mvmul.cpp" $DEVICE -ldl
g++ $HOSTFLAGS -o "$OUT/probe_dst" "$ROOT/host/probe_dst.cpp" $DEVICE -ldl

if [ -f "$ROOT/host/scale.cpp" ]; then
    g++ $HOSTFLAGS -o "$OUT/scale" "$ROOT/host/scale.cpp" $DEVICE \
        "$ROOT/host/multicore.cpp" -ldl
fi

if [ -f "$ROOT/host/check_pipeline.cpp" ]; then
    g++ $HOSTFLAGS -o "$OUT/check_pipeline" "$ROOT/host/check_pipeline.cpp" $DEVICE \
        "$ROOT/host/pipeline_matmul.cpp" -ldl
fi

if [ -f "$ROOT/host/check_rfft2.cpp" ]; then
    g++ $HOSTFLAGS -o "$OUT/check_rfft2" "$ROOT/host/check_rfft2.cpp" $DEVICE $FFT -ldl
fi

if [ -f "$ROOT/host/sweep.cpp" ]; then
    g++ $HOSTFLAGS -o "$OUT/sweep" "$ROOT/host/sweep.cpp" $DEVICE $FFT -ldl
fi
echo "built into $OUT"
