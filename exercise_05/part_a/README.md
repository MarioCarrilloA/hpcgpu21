# Exercise 5 - Part A

## 1. HOWTO

We can build and execute our code with default values  by following the next
steps:

```
./access_to_node.sh
make
./exercise05
```

The program includes **m=3, n=3** and **k=2** as default values for matrix
dimensions. However, it is possible to change them by using the flags:
**-m. -n, -k** and **-a, -b** for scalars.

Example:

```
./exercise05 -m 4096 -n 4096 -k 1024
```

In addition, you can uncomment 2 print lines inside the code for testing. These
lines produce an output as Python format. This can be used to compare the results
from CuBlas and python.

Example:

```
./exercise05 > test.py
python test.py
```
