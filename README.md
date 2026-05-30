[![CI](https://github.com/sbromberger/gompi/actions/workflows/ci.yml/badge.svg)](https://github.com/sbromberger/gompi/actions/workflows/ci.yml)

GoMPI: Message Passing Interface for Parallel Computing

The `gompi` package is a lightweight wrapper to the [OpenMPI](https://www.open-mpi.org) C library designed to develop algorithms for parallel computing.

GoMPI is a fork of the [gosl](https://github.com/cpmech/gosl) MPI library with additional methods.

## Dependencies
**This package will not work on Windows systems.**

GoMPI requires the [OpenMPI](https://www.open-mpi.org) libraries, header files, and binaries to be installed on your system.


## Testing
Testing requires four MPI ranks and is launched via `mpirun`:

```
mpirun -n 4 --oversubscribe go test .
```

## Performance
Note: latency benchmarks updated May 2026.

OSU MPI Latency Test (v7.5.1) benchmarks run using `mpirun -n 2 ./osu_latency -i 1000 -x 200` with datatype = `MPI_Char`.
GoMPI benchmarks run using `mpirun -n 2 go run latency.go`.

Benchmarks were run on a single node. Small-message overhead reflects CGo call latency and converges to parity as message size increases.

| message size (bytes) | GoMPI (µs) | OSU MPI (µs) | difference |
|---|---|---|---|
| 1 | 0.13 | 0.10 | 1.3x |
| 2 | 0.13 | 0.10 | 1.3x |
| 4 | 0.14 | 0.10 | 1.4x |
| 8 | 0.13 | 0.10 | 1.3x |
| 16 | 0.13 | 0.10 | 1.3x |
| 32 | 0.14 | 0.10 | 1.4x |
| 64 | 0.15 | 0.11 | 1.4x |
| 128 | 0.17 | 0.11 | 1.5x |
| 256 | 0.17 | 0.14 | 1.2x |
| 512 | 0.25 | 0.20 | 1.2x |
| 1024 | 0.28 | 0.23 | 1.2x |
| 2048 | 0.31 | 0.29 | 1.1x |
| 4096 | 0.66 | 0.68 | 1.0x |
| 8192 | 0.88 | 0.91 | 1.0x |
| 16384 | 1.08 | 1.15 | 0.9x |
| 32768 | 1.54 | 1.59 | 1.0x |
| 65536 | 2.92 | 2.27 | 1.3x |
| 131072 | 4.17 | 4.10 | 1.0x |
| 262144 | 7.45 | 6.83 | 1.1x |
| 524288 | 14.00 | 13.22 | 1.1x |
| 1048576 | 26.63 | 24.71 | 1.1x |
| 2097152 | 52.03 | 50.45 | 1.0x |
| 4194304 | 100.70 | 102.49 | 1.0x |

Benchmark code may be found in `cmd/latency/latency.go`.
