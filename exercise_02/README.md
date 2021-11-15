# Exercise 2

## General Properties

### HOWTO

We can build and execute binary that help us to get GPU information with the
next instructions and execute

```
./access_to_node.sh
make extras
./DevProperties
```

Now, we can calculate then the **total** number of registers of the GPU and the
number of `float`s that can be fit in shared memory by interpreting the output.
For example, below some lines from the output about the GPU that we used for
this exercise  **P2000** GPU.

```
Total amount of shared memory per block:       49152 bytes
( 8) Multiprocessors, (128) CUDA Cores/MP:     1024 CUDA Cores
Total number of registers available per block: 65536
Maximum number of threads per multiprocessor:  2048
Maximum number of threads per block:           1024
```



Now a CUDA **block** is a group of threads, in this case 1024. If our *Maximum
number of threads per multiprocessor* is 2048, it means that we have **2 blocks
per multiprocessor**.

Also, the output tells us the *Total number of registers available per block*,
so if we have 2 blocks, then we have **131072 registers per multiprocessor**.

Finally, As we have 8 *Multiprocessors*, then we have **8 x 131072 = 1048576**
registers. If we compare this for example with [Zen 2 - Microarchitectures - AMD][amd]
has a much smaller number of registers, **180** according to the documentation.

In addition, the number of floats that can fit in *shared memory* is:
**(49152/4) 4 = 3072** per block.

## Build and execute  exercise 02

We just need to follow these instructions:

```
./access_to_node.sh
make
./exercise02
```




[amd]: https://en.wikichip.org/wiki/amd/microarchitectures/zen_2
