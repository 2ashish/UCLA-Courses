
# Hash Hash Hash

Hash table implementation safe to use concurrently.

## Building

Executables can be made by running `make` from terminal, after making necessary changes in cpp files. This will create an executable with name **hash-table-tester**

## Running

Executable can be run using following command from terminal:
 `./hash-table-tester -t 6 -s 80000`
 Running above command gives below sample output:
> Generation: 94,408 usec
Hash table base: 1,467,625 usec
  \- 0 missing
Hash table v1: 2,804,374 usec
  \- 0 missing
Hash table v2: 377,155 usec
  \- 0 missing


## First Implementation

For v1, single global mutex is used which is initialized in **hash_table_v1_create()** and is destroyed in **hash_table_v1_destroy()**.
Using this mutex, the whole function **hash_table_v1_add_entry()** is locked.
This is correct as the critical section lies in the **hash_table_v1_add_entry()** function and the mutex will only allow one thread to access that region, so it will avoid any dataraces.


### Performance

Using VM with 6 cores,
 Running `./hash-table-tester -t 6 -s 80000` print output:
> Generation: 94,408 usec
Hash table base: 1,467,625 usec
  \- 0 missing
Hash table v1: 2,804,374 usec
  \- 0 missing
Hash table v2: 377,155 usec
  \- 0 missing
  
Running `./hash-table-tester -t 2 -s 240000` print output:
>Generation: 96,107 usec
Hash table base: 1,408,610 usec
  \- 0 missing
Hash table v1: 2,006,034 usec
  \- 0 missing
Hash table v2: 967,606 usec
  \- 0 missing

For Hash table v1,
With `-t 2` , 2 threads we see that v1 is  taking 142% of the base time
With `-t 6` , 6 threads we see that v1 is  taking 191% of the base time

Since we have locked the whole **hash_table_v1_add_entry()** function, even if we use multiple thread, only one thread is executing that code, so it cannot be better than serial base implementation. Also there is overhead related to the use of threads, as other threads are just waiting until the thread that has the lock gives up the lock. This becomes worse with more number of threads and we see the performance getting worse with 6 threads.

## Second Implementation

For v2, multiple mutexes are used, one for each hash table entry.
They are defined inside hash_table_entry struct and initialized in **hash_table_v2_create()** and  are destroyed in **hash_table_v2_destroy()**.
Inside **hash_table_v2_add_entry()**, we first find the hash table entry corresponding to the input key, then lock the remaining code using corresponding mutex. This way only a particular table entry is locked and other threads are free to read/write other table entry.
This is correct as only one thread is in critical section for a particular table entry. Thus, this will avoid any dataraces. Also, this will perform best as other threads are free to work parallelly on table entries that are unlocked. 

### Performance

Using VM with 6 cores,
 Running `./hash-table-tester -t 6 -s 80000` print output:
> Generation: 94,408 usec
Hash table base: 1,467,625 usec
  \- 0 missing
Hash table v1: 2,804,374 usec
  \- 0 missing
Hash table v2: 377,155 usec
  \- 0 missing
  
Running `./hash-table-tester -t 2 -s 240000` print output:
>Generation: 96,107 usec
Hash table base: 1,408,610 usec
  \- 0 missing
Hash table v1: 2,006,034 usec
  \- 0 missing
Hash table v2: 967,606 usec
  \- 0 missing

For Hash table v2,
With `-t 2` , 2 threads we see that v2 is  taking 69% of the base time
With `-t 6` , 6 threads we see that v2 is  taking 26% of the base time

We see a performance increase because in v2 threads are free to work parallelly on table entries that are unlocked, while in v1 only one thread was working at a time. We also see that increasing number of threads, v2 performance increases, showing more parallelism.

## Cleaning up

Run following command to clean all the executables,
`make clean`

> Written with [StackEdit](https://stackedit.io/).