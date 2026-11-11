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

// The coprocessor runs three independent instruction streams, called Tensix threads, and a baby
// core reaches one by the address it stores to.  T0/T1/T2 each have exactly one thread they can
// push to, and it is always TENSIX_INST_BASE from their point of view; only RISCV B can pick,
// by using the other two ports.  RISCV NC cannot push Tensix instructions at all, which is why
// a "reader kernel" moves data over the NoC rather than through the coprocessor.
#define TENSIX_INST1_BASE   0xFFE50000u   // RISCV B -> Tensix thread T1; illegal from T0/T1/T2
#define TENSIX_INST2_BASE   0xFFE60000u   // RISCV B -> Tensix thread T2; illegal from T0/T1/T2

// Each baby core sees its own 4 KiB of local data RAM at this same address, so every core can
// put its stack at the top of it without colliding with any other core.  The region above it
// is unmapped: a stack pointer past 0xFFB01000 writes nowhere, silently.
#define MEM_LOCAL_BASE      0xFFB00000u
#define MEM_LOCAL_TOP       0xFFB01000u

// -- the tile's registers, as seen from the host through a BAR window ------------------------
#define TILE_REG_WINDOW_BASE 0xFFA00000ull
#define TILE_SOFT_RESET_ADDR 0xFFB121B0u
#define BRISC_SOFT_RESET_BIT 0x800u
#define TRISC0_SOFT_RESET_BIT 0x1000u
#define TRISC1_SOFT_RESET_BIT 0x2000u
#define TRISC2_SOFT_RESET_BIT 0x4000u
#define ALL_TRISC_RESET_BITS  0x7000u
#define ALL_RISCV_RESET_BITS  0x47800u    // B, T0, T1, T2, NC

// Auto TTSync interlocks a TRISC's plain stores to Tensix configuration against the instructions
// that thread has in flight, which is the difference between a correct raw store and a race.  It
// covers T0/T1/T2 only, never B or NC, and it has to be taken out of reset to do anything.
#define AUTO_TTSYNC_SOFT_RESET_BIT 0x800000u

// Where each TRISC starts fetching when its reset bit is released.  These are the fixed values
// used unless RISCV_DEBUG_REG_TRISC_RESET_PC_OVERRIDE redirects them, and they are far enough
// apart to hold a small program each.  The command block at 0x08000 sits in the gap after T0.
#define TRISC0_RESET_PC 0x06000u
#define TRISC1_RESET_PC 0x0A000u
#define TRISC2_RESET_PC 0x0E000u

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

// RMWCIB writes one byte of a configuration register, with the value as an immediate, and it is
// a Tensix instruction rather than a store from the RISC-V core.  That difference is the whole
// point of it: an instruction is ordered against the other instructions in the same stream, so
// a base address written this way is in place for the instruction that follows and cannot be
// overwritten early by a core that has run ahead.  Four of them cover a 32-bit register.
#define TT_OP_RMWCIB0(mask, data, cfg_reg_addr) \
    TT_OP(0xb3, (((uint32_t)(mask) << 16) + ((uint32_t)(data) << 8) + (uint32_t)(cfg_reg_addr)))
#define TT_OP_RMWCIB1(mask, data, cfg_reg_addr) \
    TT_OP(0xb4, (((uint32_t)(mask) << 16) + ((uint32_t)(data) << 8) + (uint32_t)(cfg_reg_addr)))
#define TT_OP_RMWCIB2(mask, data, cfg_reg_addr) \
    TT_OP(0xb5, (((uint32_t)(mask) << 16) + ((uint32_t)(data) << 8) + (uint32_t)(cfg_reg_addr)))
#define TT_OP_RMWCIB3(mask, data, cfg_reg_addr) \
    TT_OP(0xb6, (((uint32_t)(mask) << 16) + ((uint32_t)(data) << 8) + (uint32_t)(cfg_reg_addr)))

// STALLWAIT takes two masks and they are easy to confuse, so both are spelled out here against
// the architecture's own bit names.  `stall_res` is the nine-bit *block* mask B0..B8: which kinds
// of instruction this thread may not start.  `wait_res` is the thirteen-bit *condition* mask
// C0..C12: what has to be quiet before the block lifts.  A wrong bit here is invisible on the
// simulator, which ignores the block mask outright and treats most conditions as already met,
// and becomes a data race on silicon.

// Block mask: what to hold back.
#define BLOCK_SYNC    0x002u   // B1  Sync Unit, so a SEMPOST waits behind this thread's work
#define BLOCK_PACK    0x004u   // B2  Packer
#define BLOCK_UNPACK  0x008u   // B3  Unpacker
#define BLOCK_THCON   0x020u   // B5  Scalar Unit (ThCon)
#define BLOCK_MATH    0x040u   // B6  Matrix Unit (FPU)
#define BLOCK_CFG     0x080u   // B7  Configuration Unit
#define BLOCK_SFPU    0x100u   // B8  Vector Unit (SFPU)

// Condition mask: what has to finish first.
#define WAIT_THCON    0x001u   // C0  Scalar Unit busy, or its memory requests outstanding
#define WAIT_UNPACK0  0x002u   // C1  this thread has an instruction in unpacker 0
#define WAIT_UNPACK1  0x004u   // C2  this thread has an instruction in unpacker 1
#define WAIT_PACK     0x008u   // C3  this thread has an instruction in the packer
#define WAIT_FPU      0x010u   // C4  this thread has an instruction in the matrix unit
#define WAIT_SRCA_FREE 0x020u  // C5  SrcA's unpack bank is not yet released to the unpackers
#define WAIT_SRCA_RDY 0x080u   // C7  SrcA's matrix bank is not yet released to the matrix unit
#define WAIT_SRCB_RDY 0x100u   // C8  SrcB's matrix bank is not yet released to the matrix unit
#define WAIT_CFG_WR   0x400u   // C10 this thread's RISCV has a config/GPR write still in flight

// -- the Sync Unit's semaphores ---------------------------------------------------------------
// Eight counting semaphores shared by the three threads, which is how one thread tells another
// that a buffer is filled or freed.  `sem_sel` is a one-hot mask, not an index.  SEMPOST adds
// one, SEMGET takes one away, and SEMWAIT latches the thread until the condition clears:
//   SEM_COND_NONZERO  keep waiting while the semaphore is zero   (a consumer waiting for data)
//   SEM_COND_NOTFULL  keep waiting while it is at its maximum    (a producer waiting for space)
#define SEM_COND_NONZERO 0x1u
#define SEM_COND_NOTFULL 0x2u
#define TT_OP_SEMINIT(max_value, init_value, sem_sel) \
    TT_OP(0xa3, (((uint32_t)(max_value) << 20) + ((uint32_t)(init_value) << 16) + \
                 ((uint32_t)(sem_sel) << 2)))
#define TT_OP_SEMPOST(sem_sel) TT_OP(0xa4, ((uint32_t)(sem_sel) << 2))
#define TT_OP_SEMGET(sem_sel)  TT_OP(0xa5, ((uint32_t)(sem_sel) << 2))
#define TT_OP_SEMWAIT(stall_res, sem_sel, wait_sem_cond) \
    TT_OP(0xa6, (((uint32_t)(stall_res) << 15) + ((uint32_t)(sem_sel) << 2) + \
                 (uint32_t)(wait_sem_cond)))

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
    // The three-thread firmware needs a launch signal the threads cannot race on.  `doorbell`
    // cannot serve: whichever thread cleared it would let the others see the same command twice.
    // So the host only ever increments `epoch`, each thread remembers the epoch it last ran, and
    // each sets its own bit in `done_mask` when it is finished with that epoch.
    volatile uint32_t epoch;
    volatile uint32_t done_mask;        // bit i set by thread i once it has finished this epoch
} CommandBlock;

// Field offsets, for the host, which writes words rather than the struct.
#define CMDF_DOORBELL  0u
#define CMDF_DONE      1u
#define CMDF_ROW_TILES 2u
#define CMDF_DEPTH_TILES 3u
#define CMDF_COL_TILES 4u
#define CMDF_PHASES    5u
#define CMDF_FORMAT    6u
#define CMDF_MVMULS    7u
#define CMDF_UNPACKS   8u
#define CMDF_PACKS     9u
#define CMDF_HEARTBEAT 10u
#define CMDF_ERROR     11u
#define CMDF_EPOCH     12u
#define CMDF_DONE_MASK 13u

// Which semaphore carries which handoff in the three-thread firmware.  One-hot, as `sem_sel`
// wants.  The counts are credits, not flags: SRC_FREE starts at the number of Src banks so the
// unpacker may run that many steps ahead of the matrix unit, and no further.
#define SEM_OPERANDS 0x01u   // unpack -> math: one k-step's operands are in Src
#define SEM_SRC_FREE 0x02u   // math -> unpack: a Src bank has been released
#define SEM_DST_FULL 0x04u   // math -> pack:   an output block is accumulated in Dst
#define SEM_DST_FREE 0x08u   // pack -> math:   a Dst half has been read out
#define SRC_BANKS 2u         // SrcA and SrcB are double banked, which is what allows overlap
#define DST_HALVES 2u        // our 8-row block fits twice over in Dst with room to spare
