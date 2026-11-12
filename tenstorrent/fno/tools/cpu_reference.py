"""The same partial 2-D transform on the CPU, to put the Blackhole roofline in context.

  .venv/bin/python tenstorrent/fno/tools/cpu_reference.py [out.csv]

The device figure in results.md is a roofline: an exact MVMUL count divided by a published issue
rate.  On its own it says nothing about whether the shape is worth putting on an accelerator at
all, because a CPU might already finish it in the time the launch takes.  This runs the identical
computation here and reports both sides against the same FLOP count.

What makes the comparison fair, and where it stays unfair:

  same algorithm      `rfft2_low_modes`, four matmuls: one stacked real row pass (cos and sin
                      share the operand, so they go as one call) and three Karatsuba matmuls for
                      the complex column pass.  Not an FFT, and nothing outside the band.
  same arithmetic     the same number of multiply-accumulates in the same order of magnitude of
                      precision.  fp32 throughout here; the device narrows operands to bf16 or
                      tf32 on the way into SrcA/SrcB and accumulates fp32, so the CPU is doing
                      strictly more work per multiply.
  same FLOP count     every rate below divides by 2*M*K*N summed over the four matmuls.  For
                      these shapes that equals the device's MVMUL count times 4096, because the
                      powers of two divide the 8x16x16 tile evenly and nothing is padded.
  NOT the same thing  the device number is a lower bound and this one is a measurement.  The
                      honest pairing is peak against peak, which is why `frac_peak` is here: it
                      says how much of any gap is the CPU being poor at small matmuls rather than
                      the device being fast.

`numpy.fft.rfft2` is timed alongside as the route someone would actually take on a CPU.  It
computes the whole spectrum and throws most of it away, which is the thing the band-limited
transform exists to avoid, so it is a reference and not a rival.
"""
import csv
import os
import sys
import time

# Must precede the numpy import: OpenBLAS reads its thread count at load time.
THREADS = os.environ.get("CPU_REF_THREADS", "")
if THREADS:
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = THREADS

import numpy as np

# The shapes host/check_rfft2.cpp checks, then four larger ones so both sides are past the
# fixed-overhead regime where a measurement says more about call cost than about arithmetic.
CASES = [
    ("half band, 16x16", 16, 16, 1, 8, 8),
    ("half band, 32x32", 32, 32, 1, 16, 16),
    ("half band, 64x64", 64, 64, 1, 32, 32),
    ("narrow band, 64x64", 64, 64, 1, 8, 8),
    ("rows wide, cols narrow", 64, 64, 1, 32, 8),
    ("rows narrow, cols wide", 64, 64, 1, 8, 32),
    ("full band on rows", 32, 32, 1, 32, 16),
    ("non-square field", 32, 64, 1, 16, 32),
    ("batch of 8", 32, 32, 8, 16, 16),
    ("batch of 8, 64x64", 64, 64, 8, 16, 16),
    ("batch of 64, 64x64", 64, 64, 64, 32, 32),
    ("batch of 64, 128x128", 128, 128, 64, 64, 64),
    ("batch of 256, 128x128", 128, 128, 256, 64, 64),
    ("batch of 64, 256x256", 256, 256, 64, 128, 128),
]


def stacked_rows(n_modes, length):
    """[cos; sin] for the lowest `n_modes` frequencies, stacked along the output axis."""
    angle = -2.0 * np.pi * np.outer(np.arange(n_modes), np.arange(length)) / length
    return np.concatenate([np.cos(angle), np.sin(angle)]).astype(np.float32)


def rfft2_low_modes(field, n_rows, n_cols, batch, row_modes, col_modes):
    """The four matmuls, in the order the device issues them.  Returns (re, im) and the FLOP."""
    rows = stacked_rows(row_modes, n_rows)
    flat = field.reshape(n_rows, n_cols * batch)
    both = rows @ flat
    row_re = both[:row_modes].reshape(row_modes, n_cols, batch)
    row_im = both[row_modes:].reshape(row_modes, n_cols, batch)

    # The axis swap the device folds into pack_tiles: the contracted axis has to lead, because
    # the matrix unit only performs Y = T @ X.
    x_re = row_re.transpose(1, 0, 2).reshape(n_cols, row_modes * batch)
    x_im = row_im.transpose(1, 0, 2).reshape(n_cols, row_modes * batch)

    col = stacked_rows(col_modes, n_cols)
    cos, sin = col[:col_modes], col[col_modes:]
    a = cos @ x_re
    b = sin @ x_im
    c = (cos + sin) @ (x_re + x_im)
    out_re = (a - b).reshape(col_modes, row_modes, batch).transpose(1, 0, 2)
    out_im = (c - a - b).reshape(col_modes, row_modes, batch).transpose(1, 0, 2)

    flop = 2 * (2 * row_modes) * n_rows * (n_cols * batch)
    flop += 3 * 2 * col_modes * n_cols * (row_modes * batch)
    return out_re, out_im, flop


def exact(field, n_rows, n_cols, batch, row_modes, col_modes):
    """Two-pass float64 reference, so the timed routine is known to compute the right thing."""
    f = field.astype(np.float64).reshape(n_rows, n_cols * batch)
    rows = -2.0 * np.pi * np.outer(np.arange(row_modes), np.arange(n_rows)) / n_rows
    re = (np.cos(rows) @ f).reshape(row_modes, n_cols, batch)
    im = (np.sin(rows) @ f).reshape(row_modes, n_cols, batch)
    cols = -2.0 * np.pi * np.outer(np.arange(col_modes), np.arange(n_cols)) / n_cols
    cos, sin = np.cos(cols), np.sin(cols)
    out_re = np.einsum("dw,awt->adt", cos, re) - np.einsum("dw,awt->adt", sin, im)
    out_im = np.einsum("dw,awt->adt", cos, im) + np.einsum("dw,awt->adt", sin, re)
    return out_re, out_im


def best_micros(call, reps=50, warm=10):
    """Fastest of `reps` runs.  The minimum, not the mean: it is the one the machine can repeat."""
    for _ in range(warm):
        call()
    best = float("inf")
    for _ in range(reps):
        started = time.perf_counter_ns()
        call()
        best = min(best, time.perf_counter_ns() - started)
    return best / 1000.0


def sgemm_peak():
    """What a large well-shaped sgemm reaches here, as the denominator for `frac_peak`.

    Measured rather than derived, so it carries the real all-core clock and the real memory
    system instead of a boost figure the part will not hold.
    """
    a = np.random.rand(4096, 4096).astype(np.float32)
    b = np.random.rand(4096, 4096).astype(np.float32)
    micros = best_micros(lambda: a @ b, reps=5, warm=2)
    return 2.0 * 4096 ** 3 / (micros * 1e-6) / 1e9


# One MVMUL is Dst[8x16] += SrcB[8x16] @ SrcA[16x16]: 2048 multiply-accumulates, 4096 FLOP.
TENSIX_GFLOPS = 4096 * 1.35e9 / 1e9
CARD_GFLOPS = TENSIX_GFLOPS * 120


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    threads = THREADS or "all"
    peak = sgemm_peak()
    print(f"threads {threads}   large sgemm {peak:,.0f} GFLOP/s   "
          f"one Tensix {TENSIX_GFLOPS:,.0f}   card {CARD_GFLOPS:,.0f}\n")

    rows = []
    rng = np.random.default_rng(7)
    for name, n_rows, n_cols, batch, row_modes, col_modes in CASES:
        field = rng.uniform(-1.0, 1.0, (n_rows, n_cols, batch)).astype(np.float32)
        got_re, got_im, flop = rfft2_low_modes(field, n_rows, n_cols, batch, row_modes, col_modes)
        ref_re, ref_im = exact(field, n_rows, n_cols, batch, row_modes, col_modes)
        scale = np.sqrt(np.sum(ref_re ** 2 + ref_im ** 2))
        error = np.sqrt(np.sum((got_re - ref_re) ** 2 + (got_im - ref_im) ** 2)) / scale

        micros = best_micros(
            lambda: rfft2_low_modes(field, n_rows, n_cols, batch, row_modes, col_modes))
        gflops = flop / (micros * 1e-6) / 1e9

        band = np.fft.rfft2(field, axes=(0, 1))[:row_modes, :col_modes]
        lib_micros = best_micros(lambda: np.fft.rfft2(field, axes=(0, 1))[:row_modes, :col_modes])

        rows.append({"case": name, "n_rows": n_rows, "n_cols": n_cols, "batch": batch,
                     "row_modes": row_modes, "col_modes": col_modes, "threads": threads,
                     "flop": flop, "mvmul_equiv": flop // 4096, "cpu_us": round(micros, 3),
                     "cpu_gflops": round(gflops, 2),
                     "frac_peak": round(gflops / peak, 4),
                     "rfft2_slice_us": round(lib_micros, 3),
                     "tensix_roofline_us": round(flop / 4096 / 1.35e9 * 1e6, 4),
                     "card_roofline_us": round(flop / 4096 / 1.35e9 / 120 * 1e6, 6),
                     "rel_err_vs_f64": f"{error:.3e}",
                     "band_shape": "x".join(str(v) for v in band.shape)})
        r = rows[-1]
        print(f"  {name:26} {n_rows:4}x{n_cols:<4} b{batch:<4} m {row_modes:3}x{col_modes:<3} "
              f"{flop/1e6:8.2f} MFLOP  cpu {micros:10.2f} us  {gflops:8.1f} GF/s  "
              f"{100*r['frac_peak']:5.1f}% peak  rfft2 {lib_micros:9.2f} us  "
              f"tensix {r['tensix_roofline_us']:8.3f} us  err {error:.1e}")

    if out_path:
        with open(out_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
