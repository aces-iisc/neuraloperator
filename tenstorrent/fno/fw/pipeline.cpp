// The same matmul as fw/kernel.cpp, split across three Tensix threads so the stages overlap.
//
// Blackhole's coprocessor runs three independent instruction streams, and the three baby cores
// T0, T1 and T2 each feed exactly one of them.  Nothing about the arithmetic changes here: the
// instructions issued, their operands and their order within a stage are the baseline's, and the
// output is expected to be bit-identical.  What changes is who issues them, which lets one stage
// run ahead of the next.
//
//   T0  unpack    L1 -> SrcA, SrcB        one 16x16 and one 8x16 tile per reduction step
//   T1  math      SrcA, SrcB -> Dst       MVMUL, once per fidelity phase, accumulating
//   T2  pack      Dst -> L1               two PACR per 8x16 output block
//
// Where the overlap comes from.  SrcA and SrcB are double banked in hardware: the unpacker
// writes one bank while the matrix unit reads the other, and the banks swap when a tile is
// marked valid and released.  So T0 can be unpacking reduction step k+1 while T1 multiplies
// step k, and the only thing needed from software is a credit so T0 cannot get more than a bank
// ahead.  That is what the semaphores below are.
//
//   SEM_OPERANDS   T0 posts a filled Src bank; T1 waits for one and takes it
//   SEM_SRC_FREE   T1 posts a released Src bank; T0 waits for one and takes it
//   SEM_DST_FULL   T1 posts a finished output block; T2 waits for one and takes it
//   SEM_DST_FREE   T2 posts a drained Dst; T1 waits for one and takes it
//
// Dst is not double buffered in this version, so SEM_DST_FREE carries a single credit and the
// pack of one block does not overlap the accumulation of the next.  Dst has 512 fp32 rows and
// a block uses eight of them, so there is room to do it; it is left out here because keeping
// Dst addressing identical to the baseline is what makes a bit-exact comparison meaningful.
//
// Build: compiled three times, once per thread, selected by -DTT_THREAD and linked at that
// thread's reset address.

#include "../include/tensix.h"

#if TT_THREAD != 0 && TT_THREAD != 1 && TT_THREAD != 2
#error "define TT_THREAD as 0, 1 or 2"
#endif

namespace {

// Auto TTSync is the hardware interlock that makes a TRISC's plain store to a configuration
// register safe against that thread's own in-flight instructions: the store stalls until the
// earlier instruction has passed the Wait Gate.  It exists only for T0/T1/T2, which is the
// strongest reason to run the stages on those cores rather than on RISCV B.
//
// It cannot be switched on here.  Enabling it means a SETC16 to TENSIX_TRISC_SYNC_*, and the
// simulator's register description has no such register, so the write is rejected outright.  On
// silicon this must be 1, and the base-address stores below depend on it; on the simulator it
// stays 0 and the ordering is simply not exercised, as nothing about instruction timing is.
#define FNO_AUTO_TTSYNC 0

inline void wr32(uint32_t addr, uint32_t value) {
    *(volatile uint32_t *)addr = value;
}

inline uint32_t rd32(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

// Each TRISC pushes to its own Tensix thread, and from inside that core the port is always the
// same address.  Only RISCV B would need to pick a port, and RISCV B issues nothing here.
inline void issue(uint32_t instruction) {
    wr32(TENSIX_INST_BASE, instruction);
}

inline void cfg(uint32_t reg, uint32_t value) {
    wr32(TENSIX_CFG_BASE + reg * 4u, value);
}

enum : uint32_t { CFG_PACK_BASE = 69u, CFG_UNPACK0_BASE = 76u, CFG_UNPACK1_BASE = 124u };

// A base address written as a plain store from the core.  Only safe before any instruction that
// reads it has been pushed, which here means the one-time configuration and nothing else.
inline void set_l1_base(uint32_t base_reg, uint32_t l1_addr) {
    cfg(base_reg, l1_addr / 16u - 1u);
}

// The same write, issued into this thread's instruction stream instead.
//
// This is the one place where three threads need something the one-core firmware did not.  With
// a single core, the core and the coprocessor advance together closely enough that a store
// followed by the instruction that reads it behaves itself.  With three, each core runs its own
// loop and gets ahead of its own instruction stream, so a store writes the base for a later
// step while the instruction for the current one is still queued.  Measured, not feared: with
// plain stores the pack stage wrote four output blocks' results to the addresses of four later
// blocks and left the first four never written at all.
//
// Four RMWCIB instructions carry the value a byte at a time, as immediates, and being
// instructions they cannot overtake the unpack or pack that reads them.  On silicon Auto TTSync
// would make the plain store safe too, but it covers only T0/T1/T2 and cannot be switched on
// here, so the ordered form is what the firmware uses on both.
inline void issue_l1_base(uint32_t base_reg, uint32_t l1_addr) {
    const uint32_t value = l1_addr / 16u - 1u;
    issue(TT_OP_RMWCIB0(0xFFu, value & 0xFFu, base_reg));
    issue(TT_OP_RMWCIB1(0xFFu, (value >> 8) & 0xFFu, base_reg));
    issue(TT_OP_RMWCIB2(0xFFu, (value >> 16) & 0xFFu, base_reg));
    issue(TT_OP_RMWCIB3(0xFFu, (value >> 24) & 0xFFu, base_reg));
}

inline uint32_t element_bytes(uint32_t) {
    return 4u;
}

CommandBlock *const cmd = (CommandBlock *)L1_COMMAND_ADDR;

// One word per stage, in the gap between the command block and the transform matrix.  The
// threads use it rather than a semaphore for the two things that happen before any Tensix
// instruction is issued: that the configuration is in place, and that an epoch has been seen.
// A semaphore cannot serve there, because SEMINIT is itself a Tensix instruction and a thread
// waiting on an uninitialised semaphore is undefined.
volatile uint32_t *const configured_epoch = (volatile uint32_t *)(L1_COMMAND_ADDR + 0x100u);

// -- stage 0: unpack -------------------------------------------------------------------------

#if TT_THREAD == 0

// The whole tile's configuration, written once per command by this thread alone.  Two threads
// writing the same configuration register would race with each other, and most of these
// registers are global rather than per thread, so one owner is the only safe arrangement.  The
// body is the baseline's, register for register, so the arithmetic cannot drift from it.
void configure(uint32_t format) {
    cfg(1, (format << 17) | (format << 21) | (FMT_FP32 << 25) | (1u << 29));

    cfg(18, 1u);
    cfg(70, 0x00000001u);
    cfg(71, 0u);
    cfg(28, 0x00000100u);

    cfg(12, (TILE_N * 4u) << 16);
    cfg(20, 0u);
    cfg(24, 0x0000FFFFu);

    cfg(180, 0u);
    cfg(181, 0u);
    cfg(182, 0u);
    cfg(183, 0u);

    // Unpacker 0 -> SrcA, from a 16x16 tile.
    cfg(64, (TILE_K << 16) | 0x10u | FMT_FP32);
    cfg(65, (1u << 16) | TILE_K);
    cfg(72, format);
    cfg(73, 0x03u);
    cfg(86, (TILE_K << 16) | TILE_K);
    cfg(50, 0x00000100u);
    cfg(84, (64u << 16) | 64u);

    // Unpacker 1 -> SrcB, from an 8x16 tile.
    cfg(112, (TILE_K << 16) | 0x10u | FMT_FP32);
    cfg(113, (1u << 16) | TILE_M);
    cfg(120, format);
    cfg(121, 0x03u);
    cfg(59, 0x00000100u);

    issue(TT_OP_ZEROSRC(0u, 0u, 1u, 3u));
}

// The part of the setup that is private to this Tensix thread rather than global to the tile.
//
// This is the division the split turns on, and getting it wrong is not subtle: the AddrMod table
// that SETC16 writes and the address counters that SETADCXX bounds both live per thread, so a
// stage that relies on another stage having set them finds them at their reset values.  A packer
// whose datum count was never set reports one datum per interface instead of sixteen.  Only the
// `cfg()` registers above are shared, which is why exactly one thread writes those.
void configure_thread() {
    issue(TT_OP_SETC16(5u, 4u));                            // address SrcA by the counters
    issue(TT_OP_SETC16(12u, 0x0000u));                      // AddrMod0 AB: nothing
    issue(TT_OP_SETADCXX(1u, TILE_K * TILE_K - 1u, 0u));    // unpacker 0: a whole SrcA tile
    issue(TT_OP_SETADCXX(2u, TILE_M * TILE_K - 1u, 0u));    // unpacker 1: a whole SrcB tile
}

// The credits every handoff starts with.  A maximum of one on the Dst pair is what keeps the
// pack of a block from overlapping the accumulation of the next; the Src pair gets one credit
// per hardware bank, which is exactly the depth of overlap the banks can support.
void init_semaphores() {
    issue(TT_OP_SEMINIT(SRC_BANKS, 0u, SEM_OPERANDS));
    issue(TT_OP_SEMINIT(SRC_BANKS, SRC_BANKS, SEM_SRC_FREE));
    issue(TT_OP_SEMINIT(1u, 0u, SEM_DST_FULL));
    issue(TT_OP_SEMINIT(1u, 1u, SEM_DST_FREE));
}

uint32_t run(uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases, uint32_t format) {
    const uint32_t bytes = element_bytes(format);
    const uint32_t t_tile_bytes = TILE_M * TILE_K * bytes;
    const uint32_t x_tile_bytes = TILE_K * TILE_N * bytes;
    uint32_t unpacks = 0;

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            issue(TT_OP_SETADCXY(3u, 0u, 0u, 0u, 0u, 10u));
            issue(TT_OP_SETADCZW(3u, 0u, 0u, 0u, 0u, 15u));

            for (uint32_t k = 0; k < depth; ++k) {
                // Take a bank credit before touching Src.  Without this the unpacker could
                // overwrite a bank the matrix unit has not read yet, which is the one way this
                // split can give a different answer from the baseline.
                //
                // Every SEMWAIT here blocks the Sync Unit as well as its own stage's unit.  The
                // block mask is what holds the following instructions back, and the SEMGET that
                // consumes the credit is itself a Sync Unit instruction: leave B1 out and the
                // get runs before the wait has cleared, taking a credit that was never there.
                issue(TT_OP_SEMWAIT(BLOCK_UNPACK | BLOCK_SYNC, SEM_SRC_FREE, SEM_COND_NONZERO));
                issue(TT_OP_SEMGET(SEM_SRC_FREE));

                issue_l1_base(CFG_UNPACK0_BASE,
                              L1_OPERAND_ADDR + (k * cols + j) * x_tile_bytes);
                issue_l1_base(CFG_UNPACK1_BASE,
                              L1_MATRIX_ADDR + (i * depth + k) * t_tile_bytes);
                issue(TT_OP_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u));
                issue(TT_OP_UNPACR(1u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u));

                // Hold the post behind this thread's own unpacks, or the matrix unit could be
                // told the operands are ready before they are.  Blocking the Sync Unit is the
                // architecture's idiom for handing a buffer to another thread.
                issue(TT_OP_STALLWAIT(BLOCK_SYNC, WAIT_UNPACK0 | WAIT_UNPACK1));
                issue(TT_OP_SEMPOST(SEM_OPERANDS));
                unpacks += 2u;
            }
        }
    }
    return unpacks;
}

#endif  // TT_THREAD == 0

// -- stage 1: math ---------------------------------------------------------------------------

#if TT_THREAD == 1

// The matmul stage selects AddrMod slots 0, 4 and 7, and they are this thread's own copies.
void configure_thread() {
    issue(TT_OP_SETC16(12u, 0x0000u));     // AddrMod0 AB: nothing
    issue(TT_OP_SETC16(28u, 0x0000u));     // AddrMod0 Dst: nothing
    issue(TT_OP_SETC16(16u, 0x8080u));     // AddrMod4 AB: rewind SrcA and SrcB
    issue(TT_OP_SETC16(32u, 0x2000u));     // AddrMod4 Dst: next fidelity phase
    issue(TT_OP_SETC16(19u, 0x8080u));     // AddrMod7 AB: release SrcA and SrcB rows
    issue(TT_OP_SETC16(35u, 0x8000u));     // AddrMod7 Dst: back to phase 0
}

uint32_t run(uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases, uint32_t format) {
    uint32_t mvmuls = 0;

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            // Wait for the packer to have finished with Dst before clearing it.
            issue(TT_OP_SEMWAIT(BLOCK_MATH | BLOCK_SYNC, SEM_DST_FREE, SEM_COND_NONZERO));
            issue(TT_OP_SEMGET(SEM_DST_FREE));
            issue(TT_OP_ZEROACC(1u, 0u, 0u, 0u, 0u));

            for (uint32_t k = 0; k < depth; ++k) {
                issue(TT_OP_SEMWAIT(BLOCK_MATH | BLOCK_SYNC, SEM_OPERANDS, SEM_COND_NONZERO));
                issue(TT_OP_SEMGET(SEM_OPERANDS));

                // The fidelity passes, exactly as the baseline issues them: every pass but the
                // last rewinds the operands and steps the phase, and the last one resets the
                // phase and releases both Src banks.
                for (uint32_t phase = 0; phase + 1u < phases; ++phase) {
                    issue(TT_OP_MVMUL(0u, 0u, 4u, 0u));
                }
                issue(TT_OP_MVMUL(3u, 0u, 7u, 0u));
                mvmuls += phases;

                // The release of the Src banks happens inside the last MVMUL, so the credit
                // goes back only once that instruction has left the matrix unit.
                issue(TT_OP_STALLWAIT(BLOCK_SYNC, WAIT_FPU));
                issue(TT_OP_SEMPOST(SEM_SRC_FREE));
            }

            // Dst is complete for this block once the matrix unit is drained.
            issue(TT_OP_STALLWAIT(BLOCK_SYNC, WAIT_FPU));
            issue(TT_OP_SEMPOST(SEM_DST_FULL));
        }
    }
    return mvmuls;
}

#endif  // TT_THREAD == 1

// -- stage 2: pack ---------------------------------------------------------------------------

#if TT_THREAD == 2

// The pack stage's own AddrMod slots, and the datum count the packer reads per interface.  That
// count is per read interface and must be sixteen: four interfaces then cover four rows, so an
// 8x16 block is two packs sharing one output tile.
void configure_thread() {
    issue(TT_OP_SETC16(37u, 0x0000u));              // pack AddrMod0: nothing
    issue(TT_OP_SETC16(38u, 0x0004u));              // pack AddrMod1: source Y += 4 rows
    issue(TT_OP_SETC16(39u, 0x0020u));              // pack AddrMod2: clear source Y and carry
    issue(TT_OP_SETADCXX(4u, TILE_N - 1u, 0u));     // sixteen datums per interface
}

uint32_t run(uint32_t rows, uint32_t depth, uint32_t cols, uint32_t phases, uint32_t format) {
    const uint32_t y_tile_bytes = TILE_M * TILE_N * 4u;
    uint32_t packs = 0;

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            issue(TT_OP_SEMWAIT(BLOCK_PACK | BLOCK_SYNC, SEM_DST_FULL, SEM_COND_NONZERO));
            issue(TT_OP_SEMGET(SEM_DST_FULL));

            issue_l1_base(CFG_PACK_BASE, L1_OUTPUT_ADDR + (i * cols + j) * y_tile_bytes);
            issue(TT_OP_SETADCXY(4u, 0u, 0u, 0u, 0u, 10u));
            issue(TT_OP_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));
            issue(TT_OP_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u));
            packs += 2u;

            // Give Dst back only once both packs have read it.
            issue(TT_OP_STALLWAIT(BLOCK_SYNC, WAIT_PACK));
            issue(TT_OP_SEMPOST(SEM_DST_FREE));
        }
    }
    return packs;
}

#endif  // TT_THREAD == 2

}  // namespace

// -- the loop every stage runs ---------------------------------------------------------------
//
// The host launches work by incrementing `epoch`, never by writing a flag a thread has to clear.
// Three threads cannot share a doorbell: whichever one cleared it would let the others see the
// same command a second time.  An epoch each thread compares against its own last-seen value has
// no such race, and the host waits for all three done bits before reading results.

extern "C" void kernel_main() {
    uint32_t seen = 0;

#if TT_THREAD == 0
    *configured_epoch = 0;
#endif

    for (;;) {
        const uint32_t epoch = cmd->epoch;
        if (epoch == seen) {
            continue;
        }
        if (cmd->doorbell == CMD_STOP) {
            cmd->done_mask = cmd->done_mask | (1u << TT_THREAD);
            return;
        }

        const uint32_t rows = cmd->row_tiles, depth = cmd->depth_tiles, cols = cmd->col_tiles;
        const uint32_t phases = cmd->fidelity_phases, format = cmd->format;

#if TT_THREAD == 0
        // This thread owns the configuration, so it sets it up before the other two issue
        // anything at all, and publishes the epoch it finished configuring for.
        configure(format);
        init_semaphores();
        __asm__ volatile("fence rw, rw" ::: "memory");
        *configured_epoch = epoch;
#else
        while (*configured_epoch != epoch) {
        }
#endif

        // Every stage sets up its own per-thread state, after the shared configuration is in
        // place and before it issues any work.
        configure_thread();

        const uint32_t issued = run(rows, depth, cols, phases, format);

#if TT_THREAD == 0
        cmd->unpacks = issued;
#elif TT_THREAD == 1
        cmd->mvmuls = issued;
        cmd->heartbeat = cmd->heartbeat + 1u;
#else
        cmd->packs = issued;
#endif
        seen = epoch;
        __asm__ volatile("fence rw, rw" ::: "memory");
        cmd->done_mask = cmd->done_mask | (1u << TT_THREAD);
    }
}
