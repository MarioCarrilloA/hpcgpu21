# HPC with GPU - Exercise 00

## HOWTO

Access to a node in interactive mode and build the binary
```
./access_to_node.sh
make
```

Execute the binary in the interactive node
```
./exercise00
./exercise00 > stdout.log
```


Or you can use `sbatch` and a job script to execute it
```
sbatch jobscript.sbatch
```
