// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpi

import (
	"fmt"
	"math"
	"testing"
)

const (
	tol    = 1e-10
	tolF32 = 1e-5
)


var opNames = [len(ops)]string{
	"sum", "min", "max", "prod",
	"land", "lor", "lxor",
	"band", "bor", "bxor",
}

// setSlice sets x[rank] = rankVal and all other elements to fillVal.
func setSlice[T goTypes](x []T, rank int, rankVal, fillVal T) {
	for i := range x {
		if i == rank {
			x[i] = rankVal
		} else {
			x[i] = fillVal
		}
	}
}

// valEqual compares two values with type-appropriate equality.
func valEqual[T goTypes](a, b T) bool {
	switch x := any(a).(type) {
	case float32:
		return math.Abs(float64(x)-float64(any(b).(float32))) <= tolF32
	case float64:
		return math.Abs(x-any(b).(float64)) <= tol
	case complex64:
		bv := any(b).(complex64)
		return math.Abs(float64(real(x))-float64(real(bv))) <= tolF32 &&
			math.Abs(float64(imag(x))-float64(imag(bv))) <= tolF32
	case complex128:
		bv := any(b).(complex128)
		return math.Abs(real(x)-real(bv)) <= tol &&
			math.Abs(imag(x)-imag(bv)) <= tol
	default:
		return a == b
	}
}

// slicesEqual compares two slices element-wise using valEqual.
func slicesEqual[T goTypes](a, b []T) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !valEqual(a[i], b[i]) {
			return false
		}
	}
	return true
}

func chkStatus(s Status, source, tag int) bool {
	return s.Source() == source && s.Tag() == tag
}

// testBcast verifies Bcast distributes {1,2,3,4} from root to all ranks.
func testBcast[T goTypes](t *testing.T, A *Communicator, root int) {
	t.Helper()
	b := make([]T, 4)
	exp := []T{T(1), T(2), T(3), T(4)}
	if A.Rank() == root {
		copy(b, exp)
	}
	A.Bcast(b, root)
	if !slicesEqual(b, exp) {
		t.Errorf("got %v, want %v", b, exp)
	}
}

// testReduceOp runs a single reduce op and checks the result at root.
// A nil or empty expRoot means the op is invalid or not checked for this type.
func testReduceOp[T goTypes](t *testing.T, A *Communicator, op Op, root int, x, expRoot []T) {
	t.Helper()
	res := make([]T, len(x))
	err := A.Reduce(res, x, op, root)
	valid := isValidDataTypeForOp[T](op)
	if err != nil {
		if valid {
			t.Errorf("unexpected error for valid op: %v", err)
		}
		return
	}
	if !valid {
		t.Errorf("no error for invalid op")
		return
	}
	if A.Rank() == root && len(expRoot) > 0 && !slicesEqual(res, expRoot) {
		t.Errorf("got %v, want %v", res, expRoot)
	}
}

// testAllreduceOp runs a single allreduce op and checks the result on all ranks.
// A nil or empty exp means the op is invalid or not checked for this type.
func testAllreduceOp[T goTypes](t *testing.T, A *Communicator, op Op, x, exp []T) {
	t.Helper()
	res := make([]T, len(x))
	err := A.Allreduce(res, x, op)
	valid := isValidDataTypeForOp[T](op)
	if err != nil {
		if valid {
			t.Errorf("unexpected error for valid op: %v", err)
		}
		return
	}
	if !valid {
		t.Errorf("no error for invalid op")
		return
	}
	if len(exp) > 0 && !slicesEqual(res, exp) {
		t.Errorf("got %v, want %v", res, exp)
	}
}

// signedResults returns expected reduce/allreduce results for signed integer types.
// The pattern is identical across int8, int16, int32, int64 since the test values
// (-1 fill, rank+1 for rank slot) produce the same relative results.
func signedResults[T interface{ int8 | int16 | int32 | int64 }]() [10][]T {
	return [10][]T{
		{-2, -1, 0, 1},   // OpSum
		{-1, -1, -1, -1}, // OpMin
		{1, 2, 3, 4},     // OpMax
		{-1, -2, -3, -4}, // OpProd
		{1, 1, 1, 1},     // OpLand
		{1, 1, 1, 1},     // OpLor
		{0, 0, 0, 0},     // OpLxor
		{1, 2, 3, 4},     // OpBand
		{-1, -1, -1, -1}, // OpBor
		{-2, -3, -4, -5}, // OpBxor
	}
}

// unsignedResults returns expected reduce/allreduce results for unsigned integer types.
// Uses MaxType (^T(0)) as the fill value.
func unsignedResults[T interface{ byte | uint16 | uint32 | uint64 }]() [10][]T {
	max := ^T(0)
	return [10][]T{
		{max - 1, max, 0, 1},                  // OpSum
		{1, 2, 3, 4},                           // OpMin
		{max, max, max, max},                   // OpMax
		{max, max - 1, max - 2, max - 3},       // OpProd
		{1, 1, 1, 1},                           // OpLand
		{1, 1, 1, 1},                           // OpLor
		{0, 0, 0, 0},                           // OpLxor
		{1, 2, 3, 4},                           // OpBand
		{max, max, max, max},                   // OpBor
		{max - 1, max - 2, max - 3, max - 4},  // OpBxor
	}
}

// floatResults returns expected reduce/allreduce results for float types.
// Logical and bitwise ops are invalid for floats; their entries are nil.
func floatResults[T interface{ float32 | float64 }]() [10][]T {
	return [10][]T{
		{-2, -1, 0, 1},   // OpSum
		{-1, -1, -1, -1}, // OpMin
		{1, 2, 3, 4},     // OpMax
		{-1, -2, -3, -4}, // OpProd
		nil, nil, nil, nil, nil, nil, // OpLand–OpBxor invalid
	}
}

func bcast(A *Communicator) func(*testing.T) {
	return func(t *testing.T) {
		root := 3
		for _, tc := range []struct {
			name string
			run  func()
		}{
			{"int8",    func() { testBcast[int8](t, A, root) }},
			{"byte",    func() { testBcast[byte](t, A, root) }},
			{"int16",   func() { testBcast[int16](t, A, root) }},
			{"uint16",  func() { testBcast[uint16](t, A, root) }},
			{"int32",   func() { testBcast[int32](t, A, root) }},
			{"uint32",  func() { testBcast[uint32](t, A, root) }},
			{"int64",   func() { testBcast[int64](t, A, root) }},
			{"uint64",  func() { testBcast[uint64](t, A, root) }},
			{"float32", func() { testBcast[float32](t, A, root) }},
			{"float64", func() { testBcast[float64](t, A, root) }},
			{"complex64",  func() { testBcast[complex64](t, A, root) }},
			{"complex128", func() { testBcast[complex128](t, A, root) }},
		} {
			t.Run(tc.name, func(*testing.T) { tc.run() })
			A.Barrier()
		}
	}
}

func reduce(A *Communicator) func(*testing.T) {
	root := 3
	return func(t *testing.T) {
		rank := int(A.Rank())

		xi8 := make([]int8, 4)
		setSlice(xi8, rank, int8(rank+1), int8(-1))
		xb := make([]byte, 4)
		setSlice(xb, rank, byte(rank+1), ^byte(0))
		xi16 := make([]int16, 4)
		setSlice(xi16, rank, int16(rank+1), int16(-1))
		xu16 := make([]uint16, 4)
		setSlice(xu16, rank, uint16(rank+1), ^uint16(0))
		xi32 := make([]int32, 4)
		setSlice(xi32, rank, int32(rank+1), int32(-1))
		xu32 := make([]uint32, 4)
		setSlice(xu32, rank, uint32(rank+1), ^uint32(0))
		xi64 := make([]int64, 4)
		setSlice(xi64, rank, int64(rank+1), int64(-1))
		xu64 := make([]uint64, 4)
		setSlice(xu64, rank, uint64(rank+1), ^uint64(0))
		xf32 := make([]float32, 4)
		setSlice(xf32, rank, float32(rank+1), float32(-1))
		xf64 := make([]float64, 4)
		setSlice(xf64, rank, float64(rank+1), float64(-1))
		xc64 := make([]complex64, 4)
		setSlice(xc64, rank, complex(float32(rank+1), float32(rank+1)/10), complex64(-1-1i))
		xc128 := make([]complex128, 4)
		setSlice(xc128, rank, complex(float64(rank+1), float64(rank+1)/10), complex128(-1-1i))

		ri8 := signedResults[int8]()
		rb := unsignedResults[byte]()
		ri16 := signedResults[int16]()
		ru16 := unsignedResults[uint16]()
		ri32 := signedResults[int32]()
		ru32 := unsignedResults[uint32]()
		ri64 := signedResults[int64]()
		ru64 := unsignedResults[uint64]()
		rf32 := floatResults[float32]()
		rf64 := floatResults[float64]()
		rc64 := [10][]complex64{
			{-2 - 2.9i, -1 - 2.8i, -2.7i, 1 - 2.6i},
			nil, nil,
			{2.2 - 1.8i, 4.4 - 3.6i, 6.6 - 5.4i, 8.8 - 7.2i},
			nil, nil, nil, nil, nil, nil,
		}
		rc128 := [10][]complex128{
			{-2 - 2.9i, -1 - 2.8i, -2.7i, 1 - 2.6i},
			nil, nil,
			{2.2 - 1.8i, 4.4 - 3.6i, 6.6 - 5.4i, 8.8 - 7.2i},
			nil, nil, nil, nil, nil, nil,
		}

		for opidx := range ops {
			op := Op(opidx)
			t.Run(opNames[opidx], func(t *testing.T) {
				for _, tc := range []struct {
					name string
					run  func()
				}{
					{"int8",    func() { testReduceOp(t, A, op, root, xi8, ri8[opidx]) }},
					{"byte",    func() { testReduceOp(t, A, op, root, xb, rb[opidx]) }},
					{"int16",   func() { testReduceOp(t, A, op, root, xi16, ri16[opidx]) }},
					{"uint16",  func() { testReduceOp(t, A, op, root, xu16, ru16[opidx]) }},
					{"int32",   func() { testReduceOp(t, A, op, root, xi32, ri32[opidx]) }},
					{"uint32",  func() { testReduceOp(t, A, op, root, xu32, ru32[opidx]) }},
					{"int64",   func() { testReduceOp(t, A, op, root, xi64, ri64[opidx]) }},
					{"uint64",  func() { testReduceOp(t, A, op, root, xu64, ru64[opidx]) }},
					{"float32", func() { testReduceOp(t, A, op, root, xf32, rf32[opidx]) }},
					{"float64", func() { testReduceOp(t, A, op, root, xf64, rf64[opidx]) }},
					{"complex64",  func() { testReduceOp(t, A, op, root, xc64, rc64[opidx]) }},
					{"complex128", func() { testReduceOp(t, A, op, root, xc128, rc128[opidx]) }},
				} {
					t.Run(tc.name, func(*testing.T) { tc.run() })
					A.Barrier()
				}
			})
		}
	}
}

func allreduce(A *Communicator) func(*testing.T) {
	return func(t *testing.T) {
		rank := int(A.Rank())

		xi8 := make([]int8, 4)
		setSlice(xi8, rank, int8(rank+1), int8(-1))
		xb := make([]byte, 4)
		setSlice(xb, rank, byte(rank+1), ^byte(0))
		xi16 := make([]int16, 4)
		setSlice(xi16, rank, int16(rank+1), int16(-1))
		xu16 := make([]uint16, 4)
		setSlice(xu16, rank, uint16(rank+1), ^uint16(0))
		xi32 := make([]int32, 4)
		setSlice(xi32, rank, int32(rank+1), int32(-1))
		xu32 := make([]uint32, 4)
		setSlice(xu32, rank, uint32(rank+1), ^uint32(0))
		xi64 := make([]int64, 4)
		setSlice(xi64, rank, int64(rank+1), int64(-1))
		xu64 := make([]uint64, 4)
		setSlice(xu64, rank, uint64(rank+1), ^uint64(0))
		xf32 := make([]float32, 4)
		setSlice(xf32, rank, float32(rank+1), float32(-1))
		xf64 := make([]float64, 4)
		setSlice(xf64, rank, float64(rank+1), float64(-1))
		xc64 := make([]complex64, 4)
		setSlice(xc64, rank, complex(float32(rank+1), float32(rank+1)/10), complex64(-1-1i))
		xc128 := make([]complex128, 4)
		setSlice(xc128, rank, complex(float64(rank+1), float64(rank+1)/10), complex128(-1-1i))

		ri8 := signedResults[int8]()
		rb := unsignedResults[byte]()
		ri16 := signedResults[int16]()
		ru16 := unsignedResults[uint16]()
		ri32 := signedResults[int32]()
		ru32 := unsignedResults[uint32]()
		ri64 := signedResults[int64]()
		ru64 := unsignedResults[uint64]()
		rf32 := floatResults[float32]()
		rf64 := floatResults[float64]()
		rc64 := [10][]complex64{
			{-2 - 2.9i, -1 - 2.8i, -2.7i, 1 - 2.6i},
			nil, nil,
			{2.2 - 1.8i, 4.4 - 3.6i, 6.6 - 5.4i, 8.8 - 7.2i},
			nil, nil, nil, nil, nil, nil,
		}
		rc128 := [10][]complex128{
			{-2 - 2.9i, -1 - 2.8i, -2.7i, 1 - 2.6i},
			nil, nil,
			{2.2 - 1.8i, 4.4 - 3.6i, 6.6 - 5.4i, 8.8 - 7.2i},
			nil, nil, nil, nil, nil, nil,
		}

		for opidx := range ops {
			op := Op(opidx)
			t.Run(opNames[opidx], func(t *testing.T) {
				for _, tc := range []struct {
					name string
					run  func()
				}{
					{"int8",    func() { testAllreduceOp(t, A, op, xi8, ri8[opidx]) }},
					{"byte",    func() { testAllreduceOp(t, A, op, xb, rb[opidx]) }},
					{"int16",   func() { testAllreduceOp(t, A, op, xi16, ri16[opidx]) }},
					{"uint16",  func() { testAllreduceOp(t, A, op, xu16, ru16[opidx]) }},
					{"int32",   func() { testAllreduceOp(t, A, op, xi32, ri32[opidx]) }},
					{"uint32",  func() { testAllreduceOp(t, A, op, xu32, ru32[opidx]) }},
					{"int64",   func() { testAllreduceOp(t, A, op, xi64, ri64[opidx]) }},
					{"uint64",  func() { testAllreduceOp(t, A, op, xu64, ru64[opidx]) }},
					{"float32", func() { testAllreduceOp(t, A, op, xf32, rf32[opidx]) }},
					{"float64", func() { testAllreduceOp(t, A, op, xf64, rf64[opidx]) }},
					{"complex64",  func() { testAllreduceOp(t, A, op, xc64, rc64[opidx]) }},
					{"complex128", func() { testAllreduceOp(t, A, op, xc128, rc128[opidx]) }},
				} {
					t.Run(tc.name, func(*testing.T) { tc.run() })
					A.Barrier()
				}
			})
		}
	}
}

func TestMPI(t *testing.T) {
	m, _ := Start()
	defer m.Stop()
	if m.WorldSize() < 4 {
		t.Fatal("These tests require 4 processors (are you running with mpirun?)\n")
	}

	A := m.NewCommunicator([]int{0, 1, 2, 3})

	t.Run("Bcast", bcast(A))
	A.Barrier()
	t.Run("Reduce", reduce(A))
	A.Barrier()
	t.Run("Allreduce", allreduce(A))
	A.Barrier()

	t.Run("SendFloat64s/RecvFloat64s", func(t *testing.T) {
		if A.Rank() == 0 {
			s := []float64{123, 123, 123, 123}
			for k := 1; k <= 3; k++ {
				A.Send(s, k, 1)
			}
		} else {
			y, s := A.Recv[float64](0, 1)
			if !chkStatus(s, 0, 1) {
				t.Errorf("unexpected status: source %d tag %d", s.Source(), s.Tag())
			}
			if !slicesEqual(y, []float64{123, 123, 123, 123}) {
				t.Errorf("got %v, want %v", y, []float64{123, 123, 123, 123})
			}
		}
	})
	A.Barrier()

	t.Run("SendInt64s/RecvInt64s", func(t *testing.T) {
		if A.Rank() == 0 {
			s := []int64{123, 123, 123, 123}
			for k := 1; k <= 3; k++ {
				A.Send(s, k, 2)
			}
		} else {
			y, s := A.Recv[int64](0, 2)
			if !slicesEqual(y, []int64{123, 123, 123, 123}) {
				t.Errorf("got %v, want %v", y, []int64{123, 123, 123, 123})
			}
			if !chkStatus(s, 0, 2) {
				t.Errorf("unexpected status: source %d tag %d", s.Source(), s.Tag())
			}
		}
	})
	A.Barrier()

	t.Run("SendInt64/RecvInt64", func(t *testing.T) {
		if A.Rank() == 0 {
			for k := 1; k <= 3; k++ {
				A.SendOne(int64(k*111), k, 3)
			}
		} else {
			res, s := A.RecvOne[int64](0, 3)
			exp := int64(111 * A.Rank())
			if res != exp {
				t.Errorf("got %d, want %d", res, exp)
			}
			if !chkStatus(s, 0, 3) {
				t.Errorf("unexpected status: source %d tag %d", s.Source(), s.Tag())
			}
		}
	})
	A.Barrier()

	t.Run("SendBytes/RecvBytes", func(t *testing.T) {
		if A.Rank() == 0 {
			for k := 1; k <= 3; k++ {
				A.Send(fmt.Appendf(nil, "Hello Rank %d!", k), k, 4)
			}
		} else {
			res := make([]byte, 13)
			exp := fmt.Sprintf("Hello Rank %d!", A.Rank())
			s := A.RecvPrealloc(res, 0, 4)
			if string(res) != exp {
				t.Errorf("got %s, want %s", res, exp)
			}
			if !chkStatus(s, 0, 4) {
				t.Errorf("unexpected status: source %d tag %d", s.Source(), s.Tag())
			}
		}
	})
	A.Barrier()

	t.Run("Probe", func(t *testing.T) {
		if A.Rank() == 3 {
			vals := []int64{1, 4, 9}
			for k := range 3 {
				A.Send(vals, k, 6)
			}
		} else {
			s := A.Probe(3, 6)
			if src := s.Source(); src != 3 {
				t.Errorf("Source: got %d, want 3", src)
			}
			if n := s.Count[int64](); n != 3 {
				t.Errorf("Count: got %d, want 3", n)
			}
		}
	})

	t.Run("Iprobe", func(t *testing.T) {
		if A.Rank() == 1 {
			if b, _ := A.Iprobe(3, 0); b {
				t.Errorf("Iprobe: got true, want false")
			}
		}
		if A.Rank() == 3 {
			vals := []int64{1, 4, 9}
			for k := range 3 {
				A.Send(vals, k, 6)
			}
		} else {
			if b, _ := A.Iprobe(3, AnyTag); !b {
				t.Errorf("Iprobe(3, AnyTag): got false, want true")
			}
			if b, _ := A.Iprobe(AnySource, 6); !b {
				t.Errorf("Iprobe(AnySource, 6): got false, want true")
			}
			if b, _ := A.Iprobe(AnySource, AnyTag); !b {
				t.Errorf("Iprobe(AnySource, AnyTag): got false, want true")
			}
			b, s := A.Iprobe(3, 6)
			if !b {
				t.Errorf("Iprobe(3, 6): got false, want true")
			}
			if src := s.Source(); src != 3 {
				t.Errorf("Source: got %d, want 3", src)
			}
			if n := s.Count[int64](); n != 3 {
				t.Errorf("Count: got %d, want 3", n)
			}
		}
	})
	A.Barrier()

	t.Run("WorldRank", func(t *testing.T) {
		if m.WorldRank() != int(A.Rank()) {
			t.Errorf("got %d, want %d", m.WorldRank(), A.Rank())
		}
	})
	A.Barrier()

	t.Run("WorldSize", func(t *testing.T) {
		if m.WorldSize() != 4 {
			t.Errorf("got %d, want 4", m.WorldSize())
		}
	})
	A.Barrier()

	t.Run("WorldTime", func(t *testing.T) {
		t1 := m.WorldTime()
		t2 := m.WorldTime()
		if t2 < t1 {
			t.Errorf("WorldTime not monotonic: %f < %f", t2, t1)
		}
	})
	A.Barrier()

	t.Run("Mrecv", func(t *testing.T) {
		if A.Rank() == 0 {
			s := []float64{1, 2, 3, 4}
			for k := 1; k <= 3; k++ {
				A.Send(s, k, 7)
			}
		} else {
			y, s := A.Mrecv[float64](0, 7)
			if !slicesEqual(y, []float64{1, 2, 3, 4}) {
				t.Errorf("got %v, want %v", y, []float64{1, 2, 3, 4})
			}
			if !chkStatus(s, 0, 7) {
				t.Errorf("unexpected status: source %d tag %d", s.Source(), s.Tag())
			}
		}
	})
}
