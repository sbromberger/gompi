[![CI](https://github.com/sbromberger/gompi/actions/workflows/ci.yml/badge.svg)](https://github.com/sbromberger/gompi/actions/workflows/ci.yml)

GoMPI: Message Passing Interface for Parallel Computing

The `gompi` package is a lightweight wrapper to the [OpenMPI](https://www.open-mpi.org) C library designed to develop algorithms for parallel computing.

GoMPI is a fork of the [gosl](https://github.com/cpmech/gosl) MPI library with additional methods.

## Usage

```go
package main

import (
	"fmt"
	"log"

	mpi "github.com/sbromberger/gompi"
)

func main() {
	m, err := mpi.Start()
	if err != nil {
		log.Fatal(err)
	}
	defer m.Stop()

	rank := m.WorldRank()
	size := m.WorldSize()

	comm := m.NewCommunicator(nil)

	if rank == 0 {
		// Rank 0 sends a slice of float64 to rank 1.
		vals := []float64{1.0, 2.0, 3.0}
		comm.Send(vals, 1, 0)
		fmt.Printf("rank 0 of %d: sent %v\n", size, vals)
	} else if rank == 1 {
		// Rank 1 receives from rank 0.
		vals, _ := comm.Recv[float64](0, 0)
		fmt.Printf("rank 1 of %d: received %v\n", size, vals)
	}
}
```

Run with:

```
mpirun -n 2 go run main.go
```

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

### Single-Node Benchmarks
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

### Inter-node Benchmarks (OmniPath/PSM2)

OSU MPI Latency Test (v5.9) benchmarks run using `srun -N 2 -n 2 --ntasks-per-node=1 ./osu_latency -i 1000 -x 200` with datatype = `MPI_Char`.
GoMPI benchmarks run using `srun -N 2 -n 2 --ntasks-per-node=1 go run main.go`.

Benchmarks were run across two nodes over OmniPath (PSM2 transport). Small-message overhead reflects CGo call latency and converges to parity as message size increases.

| message size (bytes) | GoMPI (µs) | OSU MPI (µs) | difference |
|---|---|---|---|
| 1 | 1.14 | 1.01 | 1.1x |
| 2 | 1.16 | 1.00 | 1.2x |
| 4 | 1.18 | 0.99 | 1.2x |
| 8 | 1.13 | 0.99 | 1.1x |
| 16 | 1.26 | 1.09 | 1.2x |
| 32 | 1.27 | 1.10 | 1.2x |
| 64 | 1.28 | 1.10 | 1.2x |
| 128 | 1.30 | 1.13 | 1.2x |
| 256 | 1.33 | 1.17 | 1.1x |
| 512 | 1.44 | 1.25 | 1.2x |
| 1024 | 1.49 | 1.37 | 1.1x |
| 2048 | 1.72 | 1.59 | 1.1x |
| 4096 | 2.15 | 2.04 | 1.1x |
| 8192 | 2.53 | 2.39 | 1.1x |
| 16384 | 7.15 | 7.03 | 1.0x |
| 32768 | 12.32 | 6.77 | 1.8x¹ |
| 65536 | 11.65 | 11.57 | 1.0x |
| 131072 | 15.80 | 15.81 | 1.0x |
| 262144 | 23.23 | 23.32 | 1.0x |
| 524288 | 36.53 | 36.22 | 1.0x |
| 1048576 | 64.27 | 64.59 | 1.0x |
| 2097152 | 119.08 | 118.28 | 1.0x |
| 4194304 | 211.46 | 212.02 | 1.0x |

¹ Anomalous result reflecting fabric instability at this message size; not representative of CGo overhead.
