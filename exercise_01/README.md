# Exercise 1

## HOWTO

### Building

To build our program
```
./access_to_node.sh
make all
```

We can build it with `fast-math` optimization flag with
```
make fast-math
```
### Execution

Every function is associated with an ID according to its desing.

| Func ID | Description                                         |
|---------|-----------------------------------------------------|
|    1    | GPU Kernel with floats                              |
|    2    | GPU Kernel with doubles                             |
|    3    | CPU host with floats                                |
|    4    | CPU host with doubles                               |
|    5    | GPU Kernel with floats, fabs instead of sqrt, pow   |
|    6    | GPU Kernel with doubles , fabs instead of sqrt, pow |

We need to specify the ID of our function with `-f` and the number of iterations
with `-n`. For example to run GPU Kernel with doubles function with 10 iteration
we need to execute.

```
./exercise01 -f 2 -n 10
```


### Execution with sbatch (All-in-one)

To execute all in one with `sbatch` we just need to use

```
sbatch  jobscript.sbatch
```

And to use `fast-math` optimization

```
sbatch --export BUILD_MODE="fast-math" jobscript.sbatch
```
**Note:** For `sbatch` we need to modify the number of iteration manually in the
script for the moment.



