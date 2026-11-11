// The tile program: one tiled matmul on Blackhole's matrix unit, driven by a command block.
//
//   Y[rows x cols] = T[rows x depth] @ X[depth x cols]
//
// This is the only shape the three transform engines need.  Each of them is one such matmul of a
// stacked cos/sin matrix against the signals, which is why the engines differ on the host and
// not here: what changes between them is T's shape and how the result is recombined, never the
// instruction stream.
//
// The matrix unit's native step is Dst[8x16] += SrcB[8x16] @ SrcA[16x16], so T arrives as 8x16
// tiles, X as 16x16 tiles, and the loop below is the obvious triple nest over them.  The
// reduction accumulates in Dst across the depth tiles, which is where fp32 accumulation happens;
// the multipliers themselves never see fp32.

#include "../include/tensix.h"

namespace {

inline void wr32(uint32_t addr, uint32_t value) {
    *(volatile uint32_t *)addr = value;
}

inline uint32_t rd32(uint32_t addr) {
    return *(volatile uint32_t *)addr;
}

// A Tensix instruction reaches the FPU by being stored to the instruction port.  The immediate
// form in the vendor headers needs an assembler extension; this store form does not, and it is
// the only form that works for an instruction whose fields are computed at runtime.
inline void issue(uint32_t instruction) {
    wr32(TENSIX_INST_BASE, instruction);
}

inline void cfg(uint32_t reg, uint32_t value) {
    wr32(TENSIX_CFG_BASE + reg * 4u, value);
}

// The unpackers and the packer address L1 through a base register holding (addr / 16) - 1.
enum : uint32_t { CFG_PACK_BASE = 69u, CFG_UNPACK0_BASE = 76u, CFG_UNPACK1_BASE = 124u };

inline void set_l1_base(uint32_t base_reg, uint32_t l1_addr) {
    cfg(base_reg, l1_addr / 16u - 1u);
}

// Bytes per element in each format the matrix unit accepts.  fp32 is stored and unpacked as
// four bytes even though the multipliers only ever see a narrower slice of the mantissa.
// L1 always holds fp32.  The unpacker narrows on the way into SrcA and SrcB, which is where
// the operand format applies: there is no separate narrow layout to maintain, and the reference
// can round the host copy the same way to keep the comparison about the arithmetic.
inline uint32_t element_bytes(uint32_t) {
    return 4u;
}

// -- one-time setup --------------------------------------------------------------------------
// Everything here is global to the tile: formats, the address-modify slots the instructions
// select between, and the counter bounds.  It is written once per command rather than per tile
// because none of it depends on the loop indices.

void configure(uint32_t format) {
    // Register numbers come from the architecture's field names, resolved with tools/cfgreg.py
    // against the simulator's own register description rather than transcribed by hand.

    // ALU_FORMAT_SPEC: the formats SrcA, SrcB and Dst carry, and fp32 accumulation in Dst.
    // Dst is fp32 whatever the operands are, which is the only wide arithmetic in the unit.
    cfg(1, (format << 17) | (format << 21) | (FMT_FP32 << 25) | (1u << 29));

    // Packer: read Dst 32 bits wide, write fp32, do not accumulate into L1.
    cfg(18, 1u);                                     // PCK_DEST_RD_CTRL read 32 bit
    cfg(70, 0x00000001u);                            // pack fp32 out from fp32 Dst, no compression
    cfg(71, 0u);                                     // no L1 accumulation
    cfg(28, 0x00000100u);                            // PACK_COUNTERS_SEC0: one read per xy plane

    // The packer masks every datum it writes against a per-column edge mask, and an unset mask
    // writes a zero instead of reading Dst.  All sixteen columns are real data here, and the
    // row-set mapping sends every row to the first mask.
    cfg(12, (TILE_N * 4u) << 16);                    // PCK0 source Y stride: one row, in bytes
    cfg(20, 0u);                                     // TILE_ROW_SET_MAPPING_0: all rows -> mask 0
    cfg(24, 0x0000FFFFu);                            // PCK_EDGE_OFFSET_SEC0_mask: all columns

    // Each of the packer's four Dst read interfaces adds its own row offset on top of the row
    // the interface index already selects.  They have to be zero or the pack reads rows the
    // matrix unit never wrote, which comes back as a block of zeros rather than as an error.
    cfg(180, 0u);
    cfg(181, 0u);
    cfg(182, 0u);
    cfg(183, 0u);

    // Unpacker 0 -> SrcA, from a 16x16 tile.
    cfg(64, (TILE_K << 16) | 0x10u | FMT_FP32);      // TileDescriptor: x_dim, uncompressed, fp32
    cfg(65, (1u << 16) | TILE_K);                    // y_dim rows in one z plane
    cfg(72, format);                                 // THCON_SEC0_REG2_Out_data_format
    cfg(73, 0x03u);                                  // zero compression off in both contexts
    cfg(86, (TILE_K << 16) | TILE_K);                // THCON_SEC0_REG5_Tile_x_dim, both contexts
    issue(TT_OP_SETC16(5u, 4u));                     // address SrcA by the counters

    // An unpack into SrcA carries a fixed four-row destination offset that the unpacker then
    // subtracts back off, so the destination base has to be exactly those 64 datums for the
    // tile to land at SrcA row 0, which is where the math side starts reading.
    cfg(50, 0x00000100u);                            // UNP0_ADD_DEST_ADDR_CNTR
    cfg(84, (64u << 16) | 64u);                      // THCON_SEC0_REG5_Dest_cntx0/1_address

    // Unpacker 1 -> SrcB, from an 8x16 tile.  SrcB takes the transform's rows, eight at a time,
    // because eight is how many output rows one MVMUL produces.
    cfg(112, (TILE_K << 16) | 0x10u | FMT_FP32);     // TileDescriptor: fp32 in L1
    cfg(113, (1u << 16) | TILE_M);                   // y_dim rows in one z plane
    cfg(120, format);                                // THCON_SEC1_REG2_Out_data_format
    cfg(121, 0x03u);                                 // zero compression off in both contexts
    cfg(59, 0x00000100u);                            // UNP1 destination Z stride, in bytes

    issue(TT_OP_ZEROSRC(0u, 0u, 1u, 3u));            // zero every bank of SrcA and SrcB

    // The AddrMod slots the loop selects between.  A slot names what happens to the operand and
    // Dst pointers after an instruction retires, which is how the fidelity sweep costs nothing
    // beyond the extra passes themselves.
    issue(TT_OP_SETC16(12u, 0x0000u));               // AddrMod0 AB: nothing
    issue(TT_OP_SETC16(28u, 0x0000u));               // AddrMod0 Dst: nothing
    issue(TT_OP_SETC16(16u, 0x8080u));               // AddrMod4 AB: rewind SrcA and SrcB
    issue(TT_OP_SETC16(32u, 0x2000u));               // AddrMod4 Dst: next fidelity phase
    issue(TT_OP_SETC16(19u, 0x8080u));               // AddrMod7 AB: release SrcA and SrcB rows
    issue(TT_OP_SETC16(35u, 0x8000u));               // AddrMod7 Dst: back to phase 0
    issue(TT_OP_SETC16(37u, 0x0000u));               // pack AddrMod0: nothing
    issue(TT_OP_SETC16(38u, 0x0004u));               // pack AddrMod1: source Y += 4 rows
    issue(TT_OP_SETC16(39u, 0x0020u));               // pack AddrMod2: clear source Y and carry

    // One unpack moves as many datums as the X counter's bounds allow, not one row, so the
    // bounds are the whole tile: a 16x16 block for SrcA and an 8x16 block for SrcB.
    issue(TT_OP_SETADCXX(1u, TILE_K * TILE_K - 1u, 0u));   // unpacker 0: a whole SrcA tile
    issue(TT_OP_SETADCXX(2u, TILE_M * TILE_K - 1u, 0u));   // unpacker 1: a whole SrcB tile
    // The packer's datum count is per read interface and must be sixteen, so one pack with all
    // four interfaces enabled moves four rows of sixteen.
    issue(TT_OP_SETADCXX(4u, TILE_N - 1u, 0u));            // sixteen datums per interface
}

// -- the loop --------------------------------------------------------------------------------

// Feed one SrcA tile and one SrcB tile, then run the fidelity passes over them.
//
// The multipliers see only part of each mantissa in one pass, so full precision is four passes
// over the same operands.  The passes are separate instructions, which makes precision a
// throughput decision: `phases` multiplies the instruction count directly.
struct Counts {
    uint32_t mvmuls;
    uint32_t unpacks;
    uint32_t packs;
};

inline uint32_t multiply_accumulate(uint32_t phases) {
    issue(TT_OP_UNPACR(0u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u));  // L1 -> SrcA
    issue(TT_OP_UNPACR(1u, 0u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u));  // L1 -> SrcB

    for (uint32_t phase = 0; phase + 1u < phases; ++phase) {
        issue(TT_OP_MVMUL(0u, 0u, 4u, 0u));          // then rewind the operands, next phase
    }
    issue(TT_OP_MVMUL(3u, 0u, 7u, 0u));              // last phase: reset fidelity, free SrcA/SrcB
    return phases;
}

Counts matmul(const CommandBlock *cmd) {
    const uint32_t rows = cmd->row_tiles, depth = cmd->depth_tiles, cols = cmd->col_tiles;
    const uint32_t phases = cmd->fidelity_phases, format = cmd->format;
    const uint32_t bytes = element_bytes(format);
    const uint32_t t_tile_bytes = TILE_M * TILE_K * bytes;     // one 8x16 block of T
    const uint32_t x_tile_bytes = TILE_K * TILE_N * bytes;     // one 16x16 block of X
    const uint32_t y_tile_bytes = TILE_M * TILE_N * 4u;        // one 8x16 block of Y, fp32

    configure(format);
    Counts counts = {0u, 0u, 0u};

    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            // One output block.  Dst starts at zero and takes the whole reduction.
            issue(TT_OP_ZEROACC(1u, 0u, 0u, 0u, 0u));
            issue(TT_OP_SETADCXY(3u, 0u, 0u, 0u, 0u, 10u));    // rewind both unpackers' Y
            issue(TT_OP_SETADCZW(3u, 0u, 0u, 0u, 0u, 15u));

            for (uint32_t k = 0; k < depth; ++k) {
                set_l1_base(CFG_UNPACK0_BASE, L1_OPERAND_ADDR + (k * cols + j) * x_tile_bytes);
                set_l1_base(CFG_UNPACK1_BASE, L1_MATRIX_ADDR + (i * depth + k) * t_tile_bytes);
                counts.mvmuls += multiply_accumulate(phases);
                counts.unpacks += 2u;              // one SrcA tile and one SrcB tile
            }

            // Drain the FPU before the packer reads Dst, and the packer before the next block
            // overwrites Dst or moves the output base.
            issue(TT_OP_STALLWAIT(BLOCK_PACK, WAIT_FPU));
            set_l1_base(CFG_PACK_BASE, L1_OUTPUT_ADDR + (i * cols + j) * y_tile_bytes);
            issue(TT_OP_SETADCXY(4u, 0u, 0u, 0u, 0u, 10u));    // packer reads from Dst row 0
            // The four Dst read interfaces cover four rows at a time, so an 8x16 block is two
            // packs: the first advances the packer's source row by four, the second puts it
            // back.  Only the second marks the tile last, so the two share one output tile and
            // the destination address carries from the first to the second instead of being
            // re-read from the base register.
            issue(TT_OP_PACR(0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u));  // rows 0..3
            issue(TT_OP_PACR(0u, 0u, 0u, 2u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u));  // rows 4..7
            issue(TT_OP_STALLWAIT(BLOCK_CFG, WAIT_PACK));
            counts.packs += 2u;                    // rows 0..3 and rows 4..7
        }
    }
    return counts;
}

}  // namespace

extern "C" void kernel_main() {
    CommandBlock *cmd = (CommandBlock *)L1_COMMAND_ADDR;
    cmd->done = CMD_IDLE;

    for (;;) {
        const uint32_t request = cmd->doorbell;
        if (request == CMD_IDLE) {
            continue;
        }
        if (request == CMD_STOP) {
            cmd->doorbell = CMD_IDLE;
            cmd->done = CMD_STOP;
            return;
        }
        cmd->error = 0;
        const Counts counts = matmul(cmd);
        cmd->mvmuls = counts.mvmuls;
        cmd->unpacks = counts.unpacks;
        cmd->packs = counts.packs;
        cmd->heartbeat = cmd->heartbeat + 1u;
        cmd->doorbell = CMD_IDLE;
        __asm__ volatile("fence rw, rw" ::: "memory");
        cmd->done = request;
    }
}
