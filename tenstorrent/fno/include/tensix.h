// Blackhole Tensix facts this port depends on: the instruction encodings it issues, the
// configuration registers it writes, and the L1 map the host and the tile agree on.
//
// Shared by the host driver and the tile firmware, so it holds no code that needs either side's
// headers.  Register numbers and instruction fields are the architecture's, not ours; they are
// transcribed from tt-isa-documentation and cross-checked against a working kernel.
#pragma once

#include <stdint.h>

// -- the tile's private windows, as seen from a RISC-V core inside it -------------------------
#define TENSIX_REGFILE_BASE 0xFFE00000u
#define TENSIX_INST_BASE    0xFFE40000u   // a 32-bit store here pushes one Tensix instruction
#define TENSIX_CFG_BASE     0xFFEF0000u   // configuration register i at +4i

// -- the tile's registers, as seen from the host through a BAR window ------------------------
#define TILE_REG_WINDOW_BASE 0xFFA00000ull
#define TILE_SOFT_RESET_ADDR 0xFFB121B0u
#define BRISC_SOFT_RESET_BIT 0x800u
#define ALL_RISCV_RESET_BITS 0x47800u

// -- instruction encoding --------------------------------------------------------------------
// Every Tensix instruction is one 32-bit word: the opcode in bits 31..24, operands below it.
// The field layouts are the architecture's.  They are transcribed from the generated encoder
// that ships with tt-lab, parameter for parameter and in its order, so a reader can diff them.
#define TT_OP(opcode, params) (((uint32_t)(opcode) << 24) + (uint32_t)(params))

// MVMUL computes Dst[8x16] += SrcB[8x16] @ SrcA[16x16]; the reduction axis is 16 long.
//   clear_dvalid  which of SrcA/SrcB to release when the instruction retires
//   instr_mod19   0 is the full matmul: eight Dst rows from eight SrcB rows.  1 is the
//                 broadcast form, which reads SrcB row 0 only and writes four Dst rows two
//                 apart -- a matrix times a vector.  Our transform needs the full form.
//   addr_mode     which AddrMod slot supplies the post-instruction pointer moves
//   dst           the Dst row the 8-row result block starts at
#define TT_OP_MVMUL(clear_dvalid, instr_mod19, addr_mode, dst) \
    TT_OP(0x26, (((uint32_t)(clear_dvalid) << 22) + ((uint32_t)(instr_mod19) << 19) + \
                 ((uint32_t)(addr_mode) << 14) + ((uint32_t)(dst) << 0)))

// UNPACR moves one tile from L1 into SrcA (block selection 0) or SrcB (1).
#define TT_OP_UNPACR(unpack_block_selection, addr_mode, cfg_context_cnt_inc, cfg_context_id, \
                     addr_cnt_context_id, ovrd_thread_id, set_dat_valid, srcb_bcast, \
                     zero_write2, auto_inc_context_id, row_search, search_cache_flush, last) \
    TT_OP(0x42, (((uint32_t)(unpack_block_selection) << 23) + ((uint32_t)(addr_mode) << 15) + \
                 ((uint32_t)(cfg_context_cnt_inc) << 13) + ((uint32_t)(cfg_context_id) << 10) + \
                 ((uint32_t)(addr_cnt_context_id) << 8) + ((uint32_t)(ovrd_thread_id) << 7) + \
                 ((uint32_t)(set_dat_valid) << 6) + ((uint32_t)(srcb_bcast) << 5) + \
                 ((uint32_t)(zero_write2) << 4) + ((uint32_t)(auto_inc_context_id) << 3) + \
                 ((uint32_t)(row_search) << 2) + ((uint32_t)(search_cache_flush) << 1) + \
                 ((uint32_t)(last) << 0)))

// PACR moves Dst rows out to L1 through the packer.
#define TT_OP_PACR(cfg_context, row_pad_zero, dst_access_mode, addr_mode, addr_cnt_context, \
                   zero_write, read_intf_sel, ovrd_thread_id, concat, ctxt_ctrl, flush, last) \
    TT_OP(0x41, (((uint32_t)(cfg_context) << 21) + ((uint32_t)(row_pad_zero) << 18) + \
                 ((uint32_t)(dst_access_mode) << 17) + ((uint32_t)(addr_mode) << 15) + \
                 ((uint32_t)(addr_cnt_context) << 13) + ((uint32_t)(zero_write) << 12) + \
                 ((uint32_t)(read_intf_sel) << 8) + ((uint32_t)(ovrd_thread_id) << 7) + \
                 ((uint32_t)(concat) << 4) + ((uint32_t)(ctxt_ctrl) << 2) + \
                 ((uint32_t)(flush) << 1) + ((uint32_t)(last) << 0)))

#define TT_OP_ZEROACC(clear_mode, use_32_bit_mode, clear_zero_flags, addr_mode, where) \
    TT_OP(0x10, (((uint32_t)(clear_mode) << 19) + ((uint32_t)(use_32_bit_mode) << 18) + \
                 ((uint32_t)(clear_zero_flags) << 17) + ((uint32_t)(addr_mode) << 14) + \
                 ((uint32_t)(where) << 0)))
#define TT_OP_ZEROSRC(zero_val, write_mode, bank_mask, src_mask) \
    TT_OP(0x11, (((uint32_t)(zero_val) << 4) + ((uint32_t)(write_mode) << 3) + \
                 ((uint32_t)(bank_mask) << 2) + ((uint32_t)(src_mask) << 0)))
#define TT_OP_SETC16(setc16_reg, setc16_value) \
    TT_OP(0xb2, (((uint32_t)(setc16_reg) << 16) + ((uint32_t)(setc16_value) << 0)))
#define TT_OP_SETADCXX(cnt_set_mask, x_end2, x_start) \
    TT_OP(0x5e, (((uint32_t)(cnt_set_mask) << 21) + ((uint32_t)(x_end2) << 10) + \
                 ((uint32_t)(x_start) << 0)))
#define TT_OP_SETADCXY(cnt_set_mask, ch1_y, ch1_x, ch0_y, ch0_x, bit_mask) \
    TT_OP(0x51, (((uint32_t)(cnt_set_mask) << 21) + ((uint32_t)(ch1_y) << 15) + \
                 ((uint32_t)(ch1_x) << 12) + ((uint32_t)(ch0_y) << 9) + \
                 ((uint32_t)(ch0_x) << 6) + ((uint32_t)(bit_mask) << 0)))
#define TT_OP_SETADCZW(cnt_set_mask, ch1_w, ch1_z, ch0_w, ch0_z, bit_mask) \
    TT_OP(0x54, (((uint32_t)(cnt_set_mask) << 21) + ((uint32_t)(ch1_w) << 15) + \
                 ((uint32_t)(ch1_z) << 12) + ((uint32_t)(ch0_w) << 9) + \
                 ((uint32_t)(ch0_z) << 6) + ((uint32_t)(bit_mask) << 0)))
#define TT_OP_SETRWC(clear_ab_vld, rwc_cr, rwc_d, rwc_b, rwc_a, bit_mask) \
    TT_OP(0x37, (((uint32_t)(clear_ab_vld) << 22) + ((uint32_t)(rwc_cr) << 18) + \
                 ((uint32_t)(rwc_d) << 14) + ((uint32_t)(rwc_b) << 10) + \
                 ((uint32_t)(rwc_a) << 6) + ((uint32_t)(bit_mask) << 0)))
#define TT_OP_INCADCZW(cnt_set_mask, ch1_y, ch1_x, ch0_y, ch0_x) \
    TT_OP(0x55, (((uint32_t)(cnt_set_mask) << 21) + ((uint32_t)(ch1_y) << 15) + \
                 ((uint32_t)(ch1_x) << 12) + ((uint32_t)(ch0_y) << 9) + ((uint32_t)(ch0_x) << 6)))
#define TT_OP_STALLWAIT(stall_res, wait_res) \
    TT_OP(0xa2, (((uint32_t)(stall_res) << 15) + ((uint32_t)(wait_res) << 0)))
#define TT_OP_NOP TT_OP(0x02, 0)

#define STALL_THCON 0x080u
#define STALL_PACK  0x100u
#define STALL_MATH  0x200u
#define WAIT_FPU    0x001u
#define WAIT_PACK   0x008u

// -- data formats the matrix unit accepts ----------------------------------------------------
// There is no fp32 multiply: fp32 in, and the unit still only multiplies the narrower forms.
#define FMT_FP32 0u
#define FMT_TF32 4u
#define FMT_BF16 5u

// -- the L1 map the host and the tile agree on -----------------------------------------------
// Blackhole gives each tile 1.5 MiB of L1.  Firmware sits at 0 because that is where BRISC
// starts fetching; everything else is placed on a 4 KiB boundary so the host can write whole
// pages without reasoning about the firmware's size.
#define L1_FIRMWARE_ADDR 0x00000u
#define L1_COMMAND_ADDR  0x08000u   // CommandBlock: what to run, and what came back
// T gets the largest share because the coupled engine's matrix is the widest thing here: a
// band of m modes costs it 2m-1 rows rather than m, doubled again for the stacked sin half.
#define L1_MATRIX_ADDR   0x10000u   // T, the transform's rows, as 8x16 SrcB tiles
#define L1_OPERAND_ADDR  0x90000u   // X, the signals, as 16x16 SrcA tiles
#define L1_OUTPUT_ADDR   0x130000u  // Y, fp32, as 8x16 Dst blocks
#define L1_SIZE          0x180000u

#define TILE_K 16u   // the reduction axis, and the width of an SrcA tile
#define TILE_M 8u    // the output rows one MVMUL produces
#define TILE_N 16u   // the signals one MVMUL produces

// What the host asks the tile to do.  One matmul of a tiled T against a tiled X.
//   Y[rows x cols] = T[rows x depth] @ X[depth x cols]
// in units of tiles: rows = row_tiles*TILE_M, depth = depth_tiles*TILE_K, cols = col_tiles*TILE_N.
#define CMD_IDLE   0u
#define CMD_MATMUL 1u
#define CMD_STOP   2u

typedef struct {
    volatile uint32_t doorbell;    // host writes a CMD_*; firmware clears it to CMD_IDLE
    volatile uint32_t done;        // firmware writes the doorbell value it finished
    volatile uint32_t row_tiles;
    volatile uint32_t depth_tiles;
    volatile uint32_t col_tiles;
    volatile uint32_t fidelity_phases;  // 1..4; how much of each mantissa the multipliers see
    volatile uint32_t format;           // FMT_* for SrcA and SrcB
    // Firmware's own count of what it issued.  Exact, not derived: the matrix unit has no
    // counter of its own, so the only true instruction count is the one kept by the code that
    // writes the instructions.  MVMUL already carries the fidelity multiplier, because a phase
    // is a separate instruction.
    volatile uint32_t mvmuls;
    volatile uint32_t unpacks;          // UNPACR: operand tiles moved L1 -> SrcA / SrcB
    volatile uint32_t packs;            // PACR: result rows moved Dst -> L1
    volatile uint32_t heartbeat;        // incremented every command, so a hang is visible
    volatile uint32_t error;
} CommandBlock;
