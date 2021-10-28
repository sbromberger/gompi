GoMPI: Message Passing Interface for Parallel Computing

The `gompi` package is a lightweight wrapper to the [OpenMPI](https://www.open-mpi.org) C++ library
designed to develop algorithms for parallel computing.

GoMPI is a fork of the [gosl](https://github.com/cpmech/gosl) MPI library with additional methods.

## Installation

1) install [OpenMPI](https://www.open-mpi.org) for your system
2) ensure  [golang.org/x/tools/cmd/stringer](https://godoc.org/golang.org/x/tools/cmd/stringer) is installed (`go get` if not)
3) run `install.bash`


## Performance

The latency benchmarks are as follows (note: the Go benchmark does not test a message size of zero bytes):

| message size(bytes) | GoMPI (us) | OSU MPI Latency Test v5.8 (us) |
|---|---|---|
| 0 | n/a  | 0.36 |
| 1 | 0.45 | 0.33 |
| 2 | 0.45 | 0.32 |
| 4 | 0.42 | 0.31 |
| 8 | 0.45 | 0.30 |
| 16 | 0.43 | 0.32 |
| 32 | 0.43 | 0.32 |
| 64 | 0.43 | 0.32 |
| 128 | 0.45 | 0.35 |
| 256 | 0.47 | 0.37 |
| 512 | 0.60 | 0.41 |
| 1024 | 0.66 | 0.34 |
| 2048 | 0.74 | 0.45 |
| 4096 | 2.01 | 1.51 |
| 8192 | 2.75 | 1.82 |
| 16384 | 3.61 | 2.36 |
| 32768 | 4.83 | 3.32 |
| 65536 | 9.24 | 4.96 |
| 131072 | 17.15 |  10.53 |
| 262144 | 31.09 | 23.07 |
| 524288 | 57.01 | 43.78 |
| 1048576 | 117.72 | 84.43 |
| 2097152 | 235.34 | 220.47 |
| 4194304 | 599.43 | 619.64 |

Benchmark code may be found in `cmd/latency.go`.
