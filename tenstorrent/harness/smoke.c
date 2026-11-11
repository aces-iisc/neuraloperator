/* Proves a built libttsim.so loads, initialises, identifies itself and advances its clock.
 * Drives the library directly through its C API, with no TT-Metalium underneath. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <libttsim.so>\n", argv[0]); return 2; }
    void *lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }

    void (*sim_init)(void)           = dlsym(lib, "libttsim_init");
    void (*sim_exit)(void)           = dlsym(lib, "libttsim_exit");
    void (*sim_clock)(uint32_t)      = dlsym(lib, "libttsim_clock");
    uint32_t (*cfg_rd)(uint32_t, uint32_t) = dlsym(lib, "libttsim_pci_config_rd32");
    if (!sim_init || !sim_exit || !sim_clock || !cfg_rd) {
        fprintf(stderr, "missing symbol: %s\n", dlerror()); return 1;
    }

    sim_init();
    /* Offset 0 of config space carries the vendor and device ID of the simulated chip. */
    uint32_t id = cfg_rd(0, 0);
    printf("vendor 0x%04x  device 0x%04x\n", id & 0xFFFF, id >> 16);

    /* How fast does simulated time advance?  This is the budget every experiment lives inside. */
    const uint32_t batch = 100000;
    for (int round = 0; round < 3; round++) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC, &a);
        sim_clock(batch);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double secs = (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        printf("%u clock steps in %.3f s  ->  %.0f steps/s\n", batch, secs, batch / secs);
    }
    sim_exit();
    printf("ok\n");
    return 0;
}
