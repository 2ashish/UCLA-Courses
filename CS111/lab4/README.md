# Hey! I'm Filing Here

Making 1MiB ext2 file system with 2 directories, 1 regular file, and 1 symbolic link.

## Building

Run code `make` to build executable

## Running


After building, run the executable using `./ext2-create` to create **cs111-base.img**
Use command `dumpe2fs cs111-base.img` to dump the filesystem information for debugging
Check the correctness of filesystem using `fsck.ext2 cs111-base.img`
Use command `mkdir mnt` to create a directory where the filesystem will be mounted via `sudo mount -o loop cs111-base.img mnt`


## Cleaning up

Filesystem can be unmounted using the command `sudo umount mnt`
The mount directory can be removed using `rmdir mnt`
Finally, all binary files can be cleaned up using command `make clean`