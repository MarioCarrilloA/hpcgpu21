# Exercise 8

## 1. HOWTO

### 1.1 Build & run

We can build and execute our code with by following the next
steps:

```
sbatch V100_info.sbatch
```

The default values for this code are the dimensions **1024x2536**

### 1.2 See results (GPU vs CPU)

At the moment the program only prints the execution time. However
it is possible to print the comparison between CUDA and CPU/Python
by uncommenting a block of easily identifiable lines of code. Then,
the ouput of `sbatch` can be used as **Python** script by typing
the next command:

```
python results-V100-<JOB_ID>.out
```

