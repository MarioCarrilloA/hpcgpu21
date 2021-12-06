# Exercise 3

## 1. HOWTO

We can build and execute our code with default values  by following the next
steps:

```
./access_to_node.sh
make
./exercise04
```

The default values are **80k** particles and **1000** kernel calls. However,
if you would like to execute with different values you can use the flag `-p` to
specify the number of particles and `-i` to specify the number of kernel calls.
E.g.

```
./exercise04 -p 5000 -i 200
```

In addition, we can build and execute the binary to get device properties by
following the next steps:

```
 make extras
 ./DevProperties
```
