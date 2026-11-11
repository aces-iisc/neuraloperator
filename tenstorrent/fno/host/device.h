// A Blackhole tile, driven from the host through ttsim's PCIe interface.
//
// The simulator is a shared library presenting a virtual PCIe device: the host forwards BAR
// accesses into it and advances its clock by hand.  Nothing here is specific to the transform,
// so this is the piece that carries over unchanged when the board arrives -- the same calls go
// to a real BAR mapping instead of to the library.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Device {
public:
    // Every Blackhole ASIC carries 140 Tensix; the cards expose 120 or 140 depending on the
    // part.  The simulator models all 140 and clocks each one that has work.
    static constexpr uint32_t kNumTensix = 140u;

    // The virtual NOC coordinate of a tile, for a 14 by 10 Tensix grid.
    static uint32_t tile_coord(uint32_t tile_id);
    static uint32_t tensix_column_x(uint32_t column);

    // The tiles that are actually usable, in id order.
    //
    // Parts are harvested: whole Tensix columns are fused off, and the ARC turns their clocks
    // off at boot, so touching one is an error rather than a slow path.  Which columns are gone
    // is read from the NOC router's column mask rather than assumed, so this follows the part
    // the simulator was configured as.  A P150 answers 120, which is what the card table says.
    const std::vector<uint32_t> &live_tiles();
    // Loads the simulator, brings up one worker tile's address windows, and leaves the tile in
    // reset.  Throws std::runtime_error on any failure the library does not treat as fatal.
    explicit Device(const std::string &library_path);
    ~Device();

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;

    // Directs every later L1, register and firmware call at one tile.
    void select_tile(uint32_t tile_id);
    uint32_t current_tile() const { return current_tile_; }

    uint32_t vendor_id() const { return vendor_id_; }
    uint32_t device_id() const { return device_id_; }

    // L1 of the worker tile, by tile-local address.
    void write_l1(uint32_t addr, const void *src, uint32_t size);
    void read_l1(uint32_t addr, void *dst, uint32_t size);
    void write_l1_u32(uint32_t addr, uint32_t value);
    uint32_t read_l1_u32(uint32_t addr);

    // The tile's register window, by the address a core inside the tile would use.
    void write_reg(uint32_t addr, uint32_t value);
    uint32_t read_reg(uint32_t addr);

    // Loads a flat firmware image at L1 address 0 and releases BRISC from reset.
    void start_firmware(const std::vector<uint8_t> &image);
    void stop_firmware();

    // Advances the simulation.  Nothing in the device moves outside these calls.
    void clock(uint32_t steps);

    // Raw Dst accumulator contents, through the tile's debug array port: two 16-bit halves of
    // row `row`, columns 2*`pair` and 2*`pair`+1.  This is the only way to see the matrix
    // unit's output without going through the packer, which is what makes it possible to tell
    // an arithmetic problem from a packing one.
    uint32_t read_dst_raw(uint32_t row, uint32_t pair);

    // Advances until `predicate` holds, in `chunk` steps at a time, up to `budget` steps.
    // Returns the steps spent, or 0 if the budget ran out.
    template <typename Predicate>
    uint64_t clock_until(Predicate predicate, uint64_t budget, uint32_t chunk = 2000) {
        for (uint64_t spent = 0; spent < budget; spent += chunk) {
            clock(chunk);
            if (predicate()) {
                return spent + chunk;
            }
        }
        return 0;
    }

private:
    void configure_window(uint32_t window, uint32_t coord, uint64_t local_addr);
    uint64_t window_base(uint32_t window) const;

    void *library_ = nullptr;
    void (*init_)() = nullptr;
    void (*exit_)() = nullptr;
    void (*clock_)(uint32_t) = nullptr;
    uint32_t (*config_rd32_)(uint32_t, uint32_t) = nullptr;
    void (*mem_rd_)(uint64_t, void *, uint32_t) = nullptr;
    void (*mem_wr_)(uint64_t, const void *, uint32_t) = nullptr;

    uint64_t bar0_ = 0;
    uint32_t current_tile_ = 0;
    std::vector<uint32_t> live_tiles_;
    uint32_t vendor_id_ = 0;
    uint32_t device_id_ = 0;
    bool running_ = false;
};
