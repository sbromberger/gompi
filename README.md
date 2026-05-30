[![CI](https://github.com/sbromberger/gompi/actions/workflows/ci.yml/badge.svg)](https://github.com/sbromberger/gompi/actions/workflows/ci.yml)

GoMPI: Message Passing Interface for Parallel Computing

The `gompi` package is a lightweight wrapper to the [OpenMPI](https://www.open-mpi.org) C++ library
designed to develop algorithms for parallel computing.

GoMPI is a fork of the [gosl](https://github.com/cpmech/gosl) MPI library with additional methods.

## Installation

1) install [OpenMPI](https://www.open-mpi.org) for your system
2) ensure  [golang.org/x/tools/cmd/stringer](https://godoc.org/golang.org/x/tools/cmd/stringer) is installed (`go install` if not)
3) run `make install`

(Other `make` options include `test`, `build`, and `clean`.)


## Performance

Note: latency benchmarks updated May 2026.

OSU MPI Latency Test (v7.5.1) bechmarks run using `mpirun -n 2 ./osu_latency -i 1000 -x 200` with datatype = `MPI_Char`.
GoMPI benchmarks run using `mpirun -n 2 go run latency.go`.

| message size (bytes) | GoMPI (µs) | OSU MPI (µs) | difference |
|---|---|---|---|
| 1 | 0.16 | 0.10 | 1.6x |
| 2 | 0.16 | 0.10 | 1.6x |
| 4 | 0.16 | 0.10 | 1.6x |
| 8 | 0.16 | 0.10 | 1.6x |
| 16 | 0.17 | 0.10 | 1.7x |
| 32 | 0.18 | 0.10 | 1.8x |
| 64 | 0.17 | 0.11 | 1.5x |
| 128 | 0.18 | 0.11 | 1.6x |
| 256 | 0.20 | 0.14 | 1.4x |
| 512 | 0.28 | 0.20 | 1.4x |
| 1024 | 0.29 | 0.23 | 1.3x |
| 2048 | 0.34 | 0.29 | 1.2x |
| 4096 | 0.69 | 0.68 | 1.0x |
| 8192 | 0.86 | 0.91 | 0.9x |
| 16384 | 1.07 | 1.15 | 0.9x |
| 32768 | 1.59 | 1.59 | 1.0x |
| 65536 | 3.08 | 2.27 | 1.4x |
| 131072 | 4.82 |  4.10 | 1.2x |
| 262144 | 8.09 | 6.83 | 1.2x |
| 524288 | 14.17 | 13.22 | 1.1x |
| 1048576 | 28.37 | 24.71 | 1.1x |
| 2097152 | 55.58 | 50.45 | 1.1x |
| 4194304 | 105.42 | 102.49 | 1.0x |

Benchmark code may be found in `cmd/latency.go`.
