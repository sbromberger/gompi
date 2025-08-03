[![CI](https://github.com/sbromberger/gompi/actions/workflows/ci.yml/badge.svg)](https://github.com/sbromberger/gompi/actions/workflows/ci.yml)

GoMPI: Message Passing Interface for Parallel Computing

The `gompi` package is a lightweight wrapper to the [OpenMPI](https://www.open-mpi.org) C++ library
designed to develop algorithms for parallel computing.

GoMPI is a fork of the [gosl](https://github.com/cpmech/gosl) MPI library with additional methods.

## Installation

1) install [OpenMPI](https://www.open-mpi.org) for your system
2) ensure  [golang.org/x/tools/cmd/stringer](https://godoc.org/golang.org/x/tools/cmd/stringer) is installed (`go get` if not)
3) run `make install`

(Other `make` options include `test`, `build`, and `clean`.)


## Performance

Note: latency benchmarks updated August 2025.

OSU bechmarks run using `mpirun -n 2 ./osu_latency -i 1000 -x 200` with datatype = `MPI_Char`.
GoMPI benchmarks run using `mpirun -n 2 go run latency.go`.

| message size (bytes) | GoMPI (µs) | OSU MPI Latency Test v7.5 (µs) |
|---|---|---|
| 1 | 0.16 | 0.11 |
| 2 | 0.18 | 0.11 |
| 4 | 0.16 | 0.11 |
| 8 | 0.17 | 0.11 |
| 16 | 0.17 | 0.11 |
| 32 | 0.17 | 0.11 |
| 64 | 0.17 | 0.12 |
| 128 | 0.19 | 0.13 |
| 256 | 0.19 | 0.16 |
| 512 | 0.28 | 0.26 |
| 1024 | 0.29 | 0.28 |
| 2048 | 0.34 | 0.35 |
| 4096 | 0.76 | 0.77 |
| 8192 | 0.92 | 0.88 |
| 16384 | 1.28 | 1.10 |
| 32768 | 1.96 | 1.45 |
| 65536 | 3.07 | 2.59 |
| 131072 | 4.48 |  4.98 |
| 262144 | 7.71 | 7.64 |
| 524288 | 13.71 | 13.25 |
| 1048576 | 25.99 | 25.25 |
| 2097152 | 56.97 | 49.65 |
| 4194304 | 140.32 | 244.79 |

Benchmark code may be found in `cmd/latency.go`.
