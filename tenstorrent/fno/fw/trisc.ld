/* One link script for all three stage images.  The base address is the reset address of the
   TRISC the image is for, passed in with --defsym FW_BASE=... , so the three builds differ by
   that one number and nothing else. */
ENTRY(_start)

SECTIONS
{
    . = FW_BASE;
    .text : { *(.text.entry) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }
    _end = .;
}
