package main

import (
	"fmt"
	"strings"

	mpi "github.com/sbromberger/gompi"
)

const (
	iterations = 1000
	warmup     = 200
	maxsize    = 1 << 22
)

func main() {
	mpi.Start(false) // line 45
	o := mpi.NewCommunicator(nil)
	if mpi.WorldSize() != 2 { // line 88
		panic("This test requires exactly 2 processors")
	}
	myId := o.Rank() // line 49

	for size := 1; size <= maxsize; size *= 2 {
		var t_total float64
		s_buf := []byte(strings.Repeat("a", size))
		r_buf := []byte(strings.Repeat("b", size))
		o.Barrier()
		for iter := 0; iter < (warmup + iterations); iter++ {
			notime := iter < warmup

			if myId == 0 {
				t_start := mpi.WorldTime() // line 140
				o.SendBytes(s_buf, 1, 1)
				o.RecvPreallocBytes(s_buf, 1, 1)
				t_end := mpi.WorldTime()
				if !notime {
					t_total += t_end - t_start
				}
			} else if myId == 1 {
				o.RecvPreallocBytes(r_buf, 0, 1)
				o.SendBytes(r_buf, 0, 1)

			}

		}
		if myId == 0 {
			latency := t_total * 1e6 / (2 * iterations)
			fmt.Printf("%-*d%*.*f\n", 10, size, 18, 2, latency)
		}
	}

	mpi.Stop()
}
