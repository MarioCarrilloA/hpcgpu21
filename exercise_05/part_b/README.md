# Exercise 5 - Part B

## 1. HOWTO

We can build and execute our code with default values  by following the next
steps:

```
./access_to_node.sh
make
./exercise05_b1
./exercise05_b2
```

The program generates two files: exercise05_b1 and exercise05_b2. B1 is to repeat part A on a Volta GPU and B2 is to repat part A with casted A and B to float16 and with a change the cublas call to tensor cores. We can also build two files one by one.

```
./access_to_node.sh
make b1
./exercise05_b1
```
```
./access_to_node.sh
make b2
./exercise05_b2
```

The program includes **m=3, n=3** and **k=2** as default values for matrix
dimensions. However, it is possible to change them by using the flags:
**-m. -n, -k** and **-a, -b** for scalars.

Example:

```
./exercise05 -m 4096 -n 4096 -k 1024
```

We can excute all together with the command:

```
salloc - -gres=gpu:V100:1 ./V100_info.sbatch
```