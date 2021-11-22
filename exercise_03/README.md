# Exercise 3

## 1. Occupancy

To answer the questions in this section it is necessary to obtain information
from the **V100** GPU. As a first step, we have to load the correct version of
the NVIDIA module to get the correct data. We searched for documentation on the
internet, and as a result, we found a [NVIDIA document][doc1] describing that
version _9.0_ should support it.

However, we decided to try more versions _(7.5, 8.0, 9.0, 10.0)_ to confirm if
we interpreted the information in the document correctly. The result of this is
that there was no difference in shared memory.
All specified, **Total amount of shared memory per block: 49152 bytes (48KB)**.
And for compute capability, **cuda capability major/minor version number: 7.0**.

Now we have the information to use the occupancy calculator.  When we choose
**7.0** as compute capability. The spreadsheet provides multiple options for
**1.b) Select Shared Memory Size Config (bytes)** The default value is **65536**,
**(64KB)**.  However, this value is above the **48KB** limit, so we chose
**32768 (32KB)**  because the sheet did not allow us to set our value directly.

Finally, according to this context, we can answer the questions. If we set the
following values we will get **100%** GPU occupancy.

- Registers Per Thread:  **32**
- Threads Per Block: **256**

Therefore we will have **Active Thread Blocks per Multiprocessor: 8** and
**32 x 256 x 8 = 65536** registers for kernel.


## 2. Little's Law

Little's Law is to see the amount of parallelism to keep ALUs always busy. It
is calculated multiplying latency and throughput.

As in the first part of this document, we tried multiple versions of the NVIDA
module _(7.5, 8.0, 9.0, 10.0)_. With the exception of the V100 GPU we obtained
the same results for all versions. The table below describes our results. The
latency units are **cycles** and for Throughput **(cores/SM)**. In addition, it
includes the needed numbers of wraps.

| GPU   | version | Latency | Throughput | Parallelism | Wraps |
|-------|---------|---------|------------|-------------|-------|
| K20Xm | ALL     | 100     | 192        | 19200       | 6     |
| P2000 | ALL     | 100     | 128        | 12800       | 4     |
| V100  | 10.0    | 100     | 64         | 6400        | 2     |
| V100  | 9.0     | 100     | 64         | 6400        | 2     |
| V100  | 8.0     | 100     | 128        | 12800       | 4     |
| V100  | 7.5     | 100     | 128        | 12800       | 4     |

A thread operates on each core and these **32** threads are grouped to warps.
To have always one warp available to be active state, the amount of throughput/32
is needed for each device.

[doc1]: https://www.nvidia.com/download/driverResults.aspx/124722/en-us
