package main

import (
	"fmt"
	"log"
	"strings"

	mpi "github.com/sbromberger/gompi/v2"
)

const (
	iterations = 1000
	warmup     = 200
	maxsize    = 1 << 22
)

func main() {
	m, err := mpi.Start()
	if err != nil {
		log.Fatal(err)
	} // line 45
	defer m.Stop()
	o := m.NewCommunicator(nil)
	if m.WorldSize() != 2 { // line 88
		panic("This test requires exactly 2 processors")
	}
	myId := o.Rank() // line 49

	for size := 1; size <= maxsize; size *= 2 {
		var t_total float64
		s_buf := []byte(strings.Repeat("a", size))
		r_buf := []byte(strings.Repeat("b", size))
		o.Barrier()
		for iter := range warmup + iterations {
			notime := iter < warmup

			switch myId {
			case 0:
				t_start := m.WorldTime() // line 140
				o.Send(s_buf, 1, 1)
				o.RecvPrealloc(s_buf, 1, 1)
				t_end := m.WorldTime()
				if !notime {
					t_total += t_end - t_start
				}
			case 1:
				o.RecvPrealloc(r_buf, 0, 1)
				o.Send(r_buf, 0, 1)

			}

		}
		o.Barrier()
		if myId == 0 {
			latency := t_total * 1e6 / (2 * iterations)
			fmt.Printf("%-*d%*.*f\n", 10, size, 18, 2, latency)
		}
	}
}
