// Each TRISC comes out of soft reset fetching from its own fixed address in L1 -- 0x06000 for
// T0, 0x0A000 for T1, 0x0E000 for T2 -- so each image is linked at that address with this stub
// first.  The three programs are otherwise the same source.
//
// The stack goes at the top of the core's own 4 KiB of local data RAM.  Every baby core sees its
// own RAM at the same address, so all three use the same value and never collide.  Note that the
// region immediately above the RAM is unmapped rather than faulting: a stack pointer set past
// the top writes nowhere and reports nothing.
.section .text.entry, "ax"
.global _start
.extern kernel_main

_start:
    li sp, 0xFFB01000
    call kernel_main
hang:
    j hang
