/* BRISC fetches from L1 address 0, so the image is linked flat at 0 with the entry first. */
ENTRY(_start)

SECTIONS
{
    . = 0;
    .text : { *(.text.entry) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) }
    .bss : { *(.bss*) *(COMMON) }
    _end = .;
}
