#include "device.h"
#include "../include/tensix.h"

#include <dlfcn.h>
#include <cstring>
#include <stdexcept>

namespace {

// BAR0 holds a table of address-translation windows.  Each entry is three 32-bit words and maps
// a 2 MiB slice of BAR0 onto one tile's address space, so reaching a tile means programming a
// window at it and then addressing the slice.
constexpr uint32_t kTlbConfigBase = 0x1FC00000u;
constexpr uint64_t kWindowBytes = 2ull * 1024 * 1024;
constexpr uint32_t kL1Window = 0;
constexpr uint32_t kRegWindow = 1;

// Blackhole lays its 140 Tensix out as 14 columns by 10 rows.  Coordinates pack as
// x | (y << 6), with two gaps the grid skips: x steps over the column at 7 (which carries
// DRAM and PCIe), and y starts at 2.  This is the simulator's own mapping, not a guess.
constexpr uint32_t kTensixColumns = 14u;

// NOC 0's register block, and the router configuration word holding the unusable-column mask.
constexpr uint32_t kNocRegsBase = 0xFFB20000u;
constexpr uint32_t kRouterCfg1 = 0x108u;

template <typename Fn>
void load(void *library, Fn &slot, const char *name) {
    slot = reinterpret_cast<Fn>(dlsym(library, name));
    if (!slot) {
        throw std::runtime_error(std::string("libttsim is missing ") + name);
    }
}

}  // namespace

Device::Device(const std::string &library_path) {
    library_ = dlopen(library_path.c_str(), RTLD_NOW);
    if (!library_) {
        throw std::runtime_error(std::string("cannot load ") + library_path + ": " + dlerror());
    }
    load(library_, init_, "libttsim_init");
    load(library_, exit_, "libttsim_exit");
    load(library_, clock_, "libttsim_clock");
    load(library_, config_rd32_, "libttsim_pci_config_rd32");
    load(library_, mem_rd_, "libttsim_pci_mem_rd_bytes");
    load(library_, mem_wr_, "libttsim_pci_mem_wr_bytes");

    init_();
    running_ = true;

    const uint32_t ids = config_rd32_(0, 0);
    vendor_id_ = ids & 0xFFFFu;
    device_id_ = ids >> 16;

    // BAR0's base, as the device advertises it.  The simulator wants the address it published,
    // not wherever a host might have mapped it.
    const uint64_t low = config_rd32_(0, 0x10) & ~0xFull;
    const uint64_t high = config_rd32_(0, 0x14);
    bar0_ = low | (high << 32);

    select_tile(0);
}

Device::~Device() {
    if (running_) {
        exit_();
    }
    if (library_) {
        dlclose(library_);
    }
}

// The Tensix columns in virtual coordinates, in order: x steps over 8 and 9, which carry DRAM
// and PCIe.  Index i is the i-th working column once the NOC translation has been applied.
uint32_t Device::tensix_column_x(uint32_t column) {
    return column + (column >= 7u ? 3u : 1u);
}

uint32_t Device::tile_coord(uint32_t tile_id) {
    const uint32_t column = tile_id % kTensixColumns;
    const uint32_t row = tile_id / kTensixColumns;
    return tensix_column_x(column) | ((row + 2u) << 6);
}

// Points both windows at one tile.  Every L1, register and firmware call afterwards goes to it.
// The host can only look at one tile at a time, but the tiles themselves run concurrently once
// started, which is what the scaling measurement needs.
void Device::select_tile(uint32_t tile_id) {
    if (tile_id >= kNumTensix) {
        throw std::runtime_error("tile id out of range");
    }
    const uint32_t coord = tile_coord(tile_id);
    configure_window(kL1Window, coord, 0);
    configure_window(kRegWindow, coord, TILE_REG_WINDOW_BASE);
    current_tile_ = tile_id;
}

// Which tiles are usable, and why the answer is not simply "all of them".
//
// A TLB window names its tile by a *virtual* coordinate, which the NOC translates before the
// request leaves the host.  That translation is what hides harvesting: the virtual Tensix
// columns are mapped onto the physical columns that survived, in order, so virtual column i is
// the i-th working column whichever ones were fused off.  Addressing a tile therefore needs no
// knowledge of which columns are gone -- only how many are left, because the virtual columns
// past that point are mapped back onto the dead ones and touching those is an error.
//
// The count comes from the NOC router's unusable-column mask rather than from a table of part
// numbers.  A P150 harvests two of the fourteen columns, leaving 12 x 10 = 120, which is the
// figure on the card.
const std::vector<uint32_t> &Device::live_tiles() {
    if (!live_tiles_.empty()) {
        return live_tiles_;
    }
    const uint32_t saved = current_tile_;
    select_tile(0);
    const uint32_t dead_columns = read_reg(kNocRegsBase + kRouterCfg1);
    select_tile(saved);

    uint32_t usable_columns = 0;
    for (uint32_t column = 0; column < kTensixColumns; ++column) {
        if (!((dead_columns >> tensix_column_x(column)) & 1u)) {
            ++usable_columns;
        }
    }
    for (uint32_t tile_id = 0; tile_id < kNumTensix; ++tile_id) {
        if (tile_id % kTensixColumns < usable_columns) {
            live_tiles_.push_back(tile_id);
        }
    }
    return live_tiles_;
}

void Device::configure_window(uint32_t window, uint32_t coord, uint64_t local_addr) {
    const uint64_t page = local_addr >> 21;
    const uint32_t words[3] = {uint32_t(page), uint32_t(page >> 32) | (coord << 11), 0};
    const uint64_t at = bar0_ + kTlbConfigBase + uint64_t(window) * 3 * 4;
    for (uint32_t i = 0; i < 3; ++i) {
        mem_wr_(at + i * 4, &words[i], sizeof(uint32_t));
    }
}

uint64_t Device::window_base(uint32_t window) const {
    return bar0_ + uint64_t(window) * kWindowBytes;
}

void Device::write_l1(uint32_t addr, const void *src, uint32_t size) {
    mem_wr_(window_base(kL1Window) + addr, src, size);
}

void Device::read_l1(uint32_t addr, void *dst, uint32_t size) {
    mem_rd_(window_base(kL1Window) + addr, dst, size);
}

void Device::write_l1_u32(uint32_t addr, uint32_t value) {
    write_l1(addr, &value, sizeof(value));
}

uint32_t Device::read_l1_u32(uint32_t addr) {
    uint32_t value = 0;
    read_l1(addr, &value, sizeof(value));
    return value;
}

void Device::write_reg(uint32_t addr, uint32_t value) {
    mem_wr_(window_base(kRegWindow) + (addr - TILE_REG_WINDOW_BASE), &value, sizeof(value));
}

uint32_t Device::read_reg(uint32_t addr) {
    uint32_t value = 0;
    mem_rd_(window_base(kRegWindow) + (addr - TILE_REG_WINDOW_BASE), &value, sizeof(value));
    return value;
}

void Device::start_firmware(const std::vector<uint8_t> &image) {
    if (image.size() > L1_COMMAND_ADDR) {
        throw std::runtime_error("firmware image overlaps the command block");
    }
    write_reg(TILE_SOFT_RESET_ADDR, ALL_RISCV_RESET_BITS);
    write_l1(L1_FIRMWARE_ADDR, image.data(), uint32_t(image.size()));

    // The command block must read as idle before the core can see it.
    std::vector<uint8_t> zeros(sizeof(CommandBlock), 0);
    write_l1(L1_COMMAND_ADDR, zeros.data(), uint32_t(zeros.size()));

    write_reg(TILE_SOFT_RESET_ADDR, ALL_RISCV_RESET_BITS & ~BRISC_SOFT_RESET_BIT);
}

void Device::stop_firmware() {
    write_reg(TILE_SOFT_RESET_ADDR, ALL_RISCV_RESET_BITS);
}

void Device::clock(uint32_t steps) {
    clock_(steps);
}

namespace {
constexpr uint32_t kDebugRegsBase = 0xFFB12000u;
constexpr uint32_t kDebugArrayReadEnable = 0x60u;
constexpr uint32_t kDebugArrayReadCommand = 0x64u;
constexpr uint32_t kDebugArrayReadData = 0x6Cu;
constexpr uint32_t kDebugArrayDst = 2u << 16;   // the command's array selector for Dst
}  // namespace

uint32_t Device::read_dst_raw(uint32_t row, uint32_t pair) {
    write_reg(kDebugRegsBase + kDebugArrayReadEnable, 1u);
    write_reg(kDebugRegsBase + kDebugArrayReadCommand, kDebugArrayDst | (pair << 12) | row);
    const uint32_t value = read_reg(kDebugRegsBase + kDebugArrayReadData);
    write_reg(kDebugRegsBase + kDebugArrayReadEnable, 0u);
    return value;
}
