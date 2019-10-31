// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpi

import (
	"fmt"
	"math"
	"os"
	"testing"
)

const tol = 1e-17

func setSliceFloat64(x []float64, rank int, offset float64) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = float64(rank+1) + offset
		} else {
			x[i] = -1
		}
	}
}

func setSliceComplex128(x []complex128, rank, ncpus int) {
	for i := 0; i < len(x); i++ {
		x[i] = 0
	}
	start, endp1 := (rank*len(x))/ncpus, ((rank+1)*len(x))/ncpus
	for i := start; i < endp1; i++ {
		x[i] = complex(float64(1+rank), float64(1+rank)/10.0)
	}
}

func setSliceInt64(x []int64, rank, ncpus int) {
	for i := 0; i < len(x); i++ {
		x[i] = -1
	}
	start, endp1 := (rank*len(x))/ncpus, ((rank+1)*len(x))/ncpus
	for i := start; i < endp1; i++ {
		x[i] = 1 + int64(rank)
	}
}

func chkArraysEqualFloat64(t *testing.T, a, b []float64) {
	if len(a) != len(b) {
		t.Errorf("arrays have different lengths (%d, %d)", len(a), len(b))
		return
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			t.Errorf("arrays do not match at index %d: %f != %f", i, a[i], b[i])
			return
		}
	}
}

func chkArraysEqualInt64(t *testing.T, a, b []int64) {
	if len(a) != len(b) {
		t.Errorf("arrays have different lengths (%d, %d)", len(a), len(b))
		return
	}
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("arrays do not match at index %d: %d != %d", i, a[i], b[i])
			return
		}
	}
}

func TestMPI(t *testing.T) {
	Start()
	defer Stop()
	if WorldSize() < 4 {
		t.Fatal("These tests require 4 processors (are you running with mpirun?)\n")
	}

	// subsets of processors
	A := NewCommunicator([]int{0, 1, 2, 3})
	if A.Rank() != 0 {
		os.Stdout, _ = os.Open(os.DevNull)
	}
	// B := NewCommunicator([]int{0, 1, 2, 3})

	// BcastDouble
	t.Run("BcastFloat64s", func(t *testing.T) {
		root := 3
		x := make([]float64, 4)
		if A.Rank() == root {
			for i := 0; i < len(x); i++ {
				x[i] = float64(1 + i)
			}
		}
		A.BcastFloat64s(x, root)
		chkArraysEqualFloat64(t, x, []float64{1, 2, 3, 4})
	})

	A.Barrier()
	// ReduceSumDouble
	t.Run("ReduceSumFloat64s", func(t *testing.T) {
		root := 3
		x := make([]float64, 4)
		setSliceFloat64(x, int(A.Rank()), 0)
		// fmt.Println("x at rank", A.Rank(), "starts as ", x)
		res := make([]float64, len(x))
		A.ReduceSumFloat64s(res, x, root)
		// fmt.Println("rank", A.Rank(), "reducesum res = ", res)
		// fmt.Println("rank", A.Rank(), "reducesum x = ", x)
		if A.Rank() == root {
			chkArraysEqualFloat64(t, res, []float64{-2, -1, 0, 1})
		} else {
			chkArraysEqualFloat64(t, res, []float64{0, 0, 0, 0})
		}
	})
	A.Barrier()

	// AllReduceSum
	t.Run("AllReduceSumFloat64s", func(t *testing.T) {
		x := make([]float64, 4)
		res := make([]float64, 4)
		setSliceFloat64(x, int(A.Rank()), 0)
		A.AllReduceSumFloat64s(res, x)
		chkArraysEqualFloat64(t, res, []float64{-2, -1, 0, 1})
	})
	A.Barrier()

	// AllReduceMin
	t.Run("AllReduceMinFloat64s", func(t *testing.T) {
		x := make([]float64, 4)
		setSliceFloat64(x, int(A.Rank()), -3.5)
		res := make([]float64, len(x))
		A.AllReduceMinFloat64s(res, x)
		// fmt.Println("allreducemin: rank", A.Rank(), "res = ", res)
		// fmt.Println("allreduceminL rank", A.Rank(), "  x = ", x)
		chkArraysEqualFloat64(t, res, []float64{-2.5, -1.5, -1, -1})
	})
	A.Barrier()

	// AllReduceMax
	t.Run("AllReduceMaxFloat64s", func(t *testing.T) {
		x := make([]float64, 4)
		setSliceFloat64(x, int(A.Rank()), 3.5)
		res := make([]float64, len(x))
		A.AllReduceMaxFloat64s(res, x)
		chkArraysEqualFloat64(t, res, []float64{4.5, 5.5, 6.5, 7.5})
	})
	A.Barrier()

	// Send & Recv
	t.Run("SendFloat64s/RecvFloat64s", func(t *testing.T) {
		if A.Rank() == 0 {
			s := []float64{123, 123, 123, 123}
			for k := 1; k <= 3; k++ {
				A.SendFloat64s(s, k, 1)
			}
		} else {
			y := A.RecvFloat64s(0, 1)
			chkArraysEqualFloat64(t, y, []float64{123, 123, 123, 123})
		}
	})
	A.Barrier()

	t.Run("SendInt64s/RecvInt64s", func(t *testing.T) {
		if A.Rank() == 0 {
			s := []int64{123, 123, 123, 123}
			for k := 1; k <= 3; k++ {
				A.SendInt64s(s, k, 2)
			}
		} else {
			y := A.RecvInt64s(0, 2)
			chkArraysEqualInt64(t, y, []int64{123, 123, 123, 123})
		}
	})

	A.Barrier()
	// SendOneI/RecvOneI
	t.Run("SendInt64/RecvInt64", func(t *testing.T) {
		if A.Rank() == 0 {
			for k := 1; k <= 3; k++ {
				A.SendInt64(int64(k*111), k, 3)
			}
		} else {
			res := A.RecvInt64(0, 3)
			exp := int64(111 * A.Rank())
			if res != exp {
				t.Errorf("received %d, expected %d", res, exp)
			}
		}
	})

	A.Barrier()
	// SendB / RecvB
	t.Run("SendBytes/RecvBytes", func(t *testing.T) {
		if A.Rank() == 0 {
			for k := 1; k <= 3; k++ {
				s := fmt.Sprintf("Hello Rank %d!", k)
				A.SendBytes([]byte(s), k, 4)
			}
		} else {
			res := make([]byte, 13)
			exp := fmt.Sprintf("Hello Rank %d!", A.Rank())
			A.RecvPreallocBytes(res, 0, 4)
			if string(res) != exp {
				t.Errorf("received %s, expected %s", res, exp)
			}
		}
	})

	A.Barrier()

	// SendOneString / RecvOneString
	t.Run("SendString/RecvString", func(t *testing.T) {
		if A.Rank() == 0 {
			for k := 1; k <= 3; k++ {
				s := fmt.Sprintf("Hello Rank %d!", k)
				A.SendString(s, k, 5)
			}
		} else {
			res := A.RecvString(0, 5)
			exp := fmt.Sprintf("Hello Rank %d!", A.Rank())
			if res != exp {
				t.Errorf("received %s, expected %s", res, exp)
			}
		}
	})

	A.Barrier()

	// Probe
	t.Run("Probe", func(t *testing.T) {
		if A.Rank() == 3 {
			vals := []int64{1, 4, 9}
			for k := 0; k < 3; k++ {
				A.SendInt64s(vals, k, 6)
			}
		} else {
			s := A.Probe(3, 6)
			src := s.GetSource()
			if src != 3 {
				t.Errorf("GetSource: received %d, expected 3", src)
			}
			n := s.GetCount(Long)
			if n != 3 {
				t.Errorf("GetCount: received %d, expected 3", n)
			}
		}
	})
}

//
// 	} else {
//
// 		// BcastFromRootC
// 		x := make([]complex128, 8)
// 		if B.Rank() == 0 {
// 			for i := 0; i < len(x); i++ {
// 				x[i] = complex(float64(1+i), float64(1+i)/10.0)
// 			}
// 		}
// 		B.BcastFromRootC(x)
// 		chk.ArrayC(tst, "B: x (complex)", 1e-17, x, []complex128{1 + 0.1i, 2 + 0.2i, 3 + 0.3i, 4 + 0.4i, 5 + 0.5i, 6 + 0.6i, 7 + 0.7i, 8 + 0.8i})
//
// 		// ReduceSum
// 		setSliceC(x, int(B.Rank()), int(B.Size()))
// 		res := make([]complex128, len(x))
// 		B.ReduceSumC(res, x)
// 		if B.Rank() == 0 {
// 			chk.ArrayC(tst, "B root: res", 1e-17, res, []complex128{1 + 0.1i, 1 + 0.1i, 2 + 0.2i, 2 + 0.2i, 3 + 0.3i, 3 + 0.3i, 4 + 0.4i, 4 + 0.4i})
// 		} else {
// 			chk.ArrayC(tst, "B others: res", 1e-17, res, nil)
// 		}
//
// 		// AllReduceSumC
// 		setSliceC(x, int(B.Rank()), int(B.Size()))
// 		for i := 0; i < len(x); i++ {
// 			res[i] = 0
// 		}
// 		B.AllReduceSumC(res, x)
// 		chk.ArrayC(tst, "B all: res", 1e-17, res, []complex128{1 + 0.1i, 1 + 0.1i, 2 + 0.2i, 2 + 0.2i, 3 + 0.3i, 3 + 0.3i, 4 + 0.4i, 4 + 0.4i})
//
// 		// AllReduceMinI
// 		z := make([]int, 8)
// 		zres := make([]int, 8)
// 		setSliceI(z, int(B.Rank()), int(B.Size()))
// 		B.AllReduceMinI(zres, z)
// 		chk.Ints(tst, "A all (min int): res", zres, []int{-1, -1, -1, -1, -1, -1, -1, -1})
//
// 		// AllReduceMaxI
// 		setSliceI(z, int(B.Rank()), int(B.Size()))
// 		for i := 0; i < len(z); i++ {
// 			zres[i] = 0
// 		}
// 		B.AllReduceMaxI(zres, z)
// 		chk.Ints(tst, "A all (max int): res", zres, []int{1, 1, 2, 2, 3, 3, 4, 4})
//
// 		// SendC & RecvC
// 		if B.Rank() == 0 {
// 			s := []complex128{123 + 1i, 123 + 2i, 123 + 3i, 123 + 4i}
// 			for k := 1; k <= 3; k++ {
// 				B.SendC(s, k)
// 			}
// 		} else {
// 			y := make([]complex128, 4)
// 			B.RecvC(y, 0)
// 			chk.ArrayC(tst, "B recv", 1e-17, y, []complex128{123 + 1i, 123 + 2i, 123 + 3i, 123 + 4i})
// 		}
//
// 		// SendOne & RecvOne
// 		if B.Rank() == 0 {
// 			for k := 1; k <= 3; k++ {
// 				B.SendOne(-123, k)
// 			}
// 		} else {
// 			res := B.RecvOne(0)
// 			chk.Float64(tst, "B RecvOne", 1e-17, res, -123)
// 		}
// 	}
//
// 	// wait for all
// 	world := mpi.NewCommunicator(nil)
// 	world.Barrier()
// }
