// BRISC starts fetching at L1 address 0 when its soft-reset bit is released, so this is the
// first thing in the image.  The stack goes at the top of BRISC's private data memory.
.section .text.entry, "ax"
.global _start
.extern kernel_main

_start:
    li sp, 0xFFB02000
    call kernel_main
hang:
    j hang
