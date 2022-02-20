# A Kernel Seedling

Kernel code to count number of running processes

## Building

make clean
make

## Running

sudo insmod proc_count.ko

## Cleaning Up

sudo rmmod proc_count

## Testing

python -m unittest

Kernel release version:
Linux 5.14.8-arch1-1 #1 SMP PREEMPT

Report which kernel release version you tested your module on
(hint: use `uname`, check for options with `man uname`).
It should match release numbers as seen on https://www.kernel.org/.

