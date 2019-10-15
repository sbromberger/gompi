#!/bin/bash

go build -o /tmp/gompi/t_mpi00_main t_mpi00_main.go && mpirun -np 4 /tmp/gompi/t_mpi00_main

tests="t_mpi01_main t_mpi02_main t_mpi03_main t_mpi04_main"
for t in $tests; do
    go build -o /tmp/gompi/$t "$t".go && mpirun -np 3 /tmp/gompi/$t
done
