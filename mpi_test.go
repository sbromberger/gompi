// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mpi

import (
	"fmt"
	"math"
	"testing"
)

const tol = 1e-17
const (
	MaxUint32 = ^uint32(0)
	MaxUint64 = ^uint64(0)
)

func setSliceByte(x []byte, rank int, offset byte) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = byte(rank+1) + offset
		} else {
			x[i] = 0xff
		}
	}
}

func setSliceUint32(x []uint32, rank int, offset uint32) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = uint32(rank+1) + offset
		} else {
			x[i] = MaxUint32
		}
	}
}

func setSliceInt32(x []int32, rank int, offset int32) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = int32(rank+1) + offset
		} else {
			x[i] = -1
		}
	}
}

func setSliceUint64(x []uint64, rank int, offset uint64) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = uint64(rank+1) + offset
		} else {
			x[i] = MaxUint64
		}
	}
}

func setSliceInt64(x []int64, rank int, offset int64) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = int64(rank+1) + offset
		} else {
			x[i] = -1
		}
	}
}

func setSliceFloat32(x []float32, rank int, offset float32) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = float32(rank+1) + offset
		} else {
			x[i] = -1
		}
	}
}
func setSliceFloat64(x []float64, rank int, offset float64) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = float64(rank+1) + offset
		} else {
			x[i] = -1
		}
	}
}

func setSliceComplex128(x []complex128, rank int, offset complex128) {
	for i := 0; i < len(x); i++ {
		if i == rank {
			x[i] = complex(float64(rank+1), float64(rank+1)/10.0) + offset
		} else {
			x[i] = complex(float64(-1), float64(-1))
		}
	}
}

func chkArraysEqualByte(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func chkArraysEqualUint32(a, b []uint32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func chkArraysEqualInt32(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func chkArraysEqualUint64(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func chkArraysEqualInt64(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func chkArraysEqualFloat32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i])-float64(b[i])) > tol {
			return false
		}
	}
	return true
}

func chkArraysEqualFloat64(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

func chkArraysEqualComplex128(a, b []complex128) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(real(a[i])-real(b[i])) > tol || math.Abs(imag(a[i])-imag(b[i])) > tol {
			return false
		}
	}
	return true
}

func bcast(A Communicator, t *testing.T) func(*testing.T) {
	return func(t *testing.T) {
		root := 3
		t.Run("byte", func(t *testing.T) {
			b := make([]byte, 4)
			if A.Rank() == root {
				for i := 0; i < len(b); i++ {
					b[i] = byte(1 + i)
				}
			}
			var exp = []byte{1, 2, 3, 4}
			A.BcastBytes(b, root)
			if !chkArraysEqualByte(b, exp) {
				t.Errorf("received %v, expected %v", b, exp)
			}
		})
		A.Barrier()

		t.Run("uint32", func(t *testing.T) {
			u32 := make([]uint32, 4)
			if A.Rank() == root {
				for i := 0; i < len(u32); i++ {
					u32[i] = uint32(1 + i)
				}
			}
			var exp = []uint32{1, 2, 3, 4}
			A.BcastUint32s(u32, root)
			if !chkArraysEqualUint32(u32, exp) {
				t.Errorf("received %v, expected %v", u32, exp)
			}
		})
		A.Barrier()

		t.Run("int32", func(t *testing.T) {
			i32 := make([]int32, 4)
			if A.Rank() == root {
				for i := 0; i < len(i32); i++ {
					i32[i] = int32(1 + i)
				}
			}
			var exp = []int32{1, 2, 3, 4}
			A.BcastInt32s(i32, root)
			if !chkArraysEqualInt32(i32, exp) {
				t.Errorf("received %v, expected %v", i32, exp)
			}

		})
		A.Barrier()

		t.Run("uint64", func(t *testing.T) {
			u64 := make([]uint64, 4)
			if A.Rank() == root {
				for i := 0; i < len(u64); i++ {
					u64[i] = uint64(1 + i)
				}
			}
			exp := []uint64{1, 2, 3, 4}
			A.BcastUint64s(u64, root)
			if !chkArraysEqualUint64(u64, exp) {
				t.Errorf("received %v, expected %v", u64, exp)
			}
		})
		A.Barrier()

		t.Run("int64", func(t *testing.T) {
			i64 := make([]int64, 4)
			if A.Rank() == root {
				for i := 0; i < len(i64); i++ {
					i64[i] = int64(1 + i)
				}
			}
			var exp = []int64{1, 2, 3, 4}
			A.BcastInt64s(i64, root)
			if !chkArraysEqualInt64(i64, exp) {
				t.Errorf("received %v, expected %v", i64, exp)
			}
		})
		A.Barrier()

		t.Run("float32", func(t *testing.T) {
			f32 := make([]float32, 4)
			if A.Rank() == root {
				for i := 0; i < len(f32); i++ {
					f32[i] = float32(1 + i)
				}
			}
			var exp = []float32{1, 2, 3, 4}
			A.BcastFloat32s(f32, root)
			if !chkArraysEqualFloat32(f32, exp) {
				t.Errorf("received %v, expected %v", f32, exp)
			}
		})
		A.Barrier()

		t.Run("float64", func(t *testing.T) {
			f64 := make([]float64, 4)
			if A.Rank() == root {
				for i := 0; i < len(f64); i++ {
					f64[i] = float64(1 + i)
				}
			}
			var exp = []float64{1, 2, 3, 4}
			A.BcastFloat64s(f64, root)
			if !chkArraysEqualFloat64(f64, exp) {
				t.Errorf("received %v, expected %v", f64, exp)
			}
		})
		A.Barrier()

		t.Run("complex128", func(t *testing.T) {
			c128 := make([]complex128, 4)
			if A.Rank() == root {
				for i := 0; i < len(c128); i++ {
					c128[i] = complex(float64(1+i), float64(i))
				}
			}
			var exp = []complex128{complex(1, 0), complex(2, 1), complex(3, 2), complex(4, 3)}
			A.BcastComplex128s(c128, root)
			if !chkArraysEqualComplex128(c128, exp) {
				t.Errorf("received %v, expected %v", c128, exp)
			}
		})
		A.Barrier()
	}
}

func reduce(A Communicator, t *testing.T) func(*testing.T) {
	root := 3
	testNames := [...]string{
		"sum",
		"min",
		"max",
		"prod",
		"land",
		"lor",
		"lxor",
		"band",
		"bor",
		"bxor",
	}

	return func(t *testing.T) {
		for opidx := range ops {
			t.Run(testNames[opidx], func(t *testing.T) {
				op := Op(opidx)
				t.Run("byte", func(t *testing.T) {
					results := [...][]byte{
						{0xfe, 0xff, 0, 1},       // sum
						{1, 2, 3, 4},             // min
						{0xff, 0xff, 0xff, 0xff}, // max
						{0xff, 0xfe, 0xfd, 0xfc}, // prod
						{1, 1, 1, 1},             // land
						{1, 1, 1, 1},             // lor
						{0, 0, 0, 0},             // lxor
						{1, 2, 3, 4},             // band
						{0xff, 0xff, 0xff, 0xff}, // bor
						{0xfe, 0xfd, 0xfc, 0xfb}, // bxor
					}

					x := make([]byte, 4)
					setSliceByte(x, int(A.Rank()), 0)
					res := make([]byte, len(x))
					err := A.ReduceBytes(res, x, op, root)
					valid := isValidDataTypeForOp(Byte, op)

					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]byte, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualByte(res, exp) {
						fmt.Println("x = ", x)
						t.Errorf("rank %d received %v, expected %v", A.Rank(), res, exp)
					}
				})
				A.Barrier()

				t.Run("uint32", func(t *testing.T) {
					results := [...][]uint32{
						{MaxUint32 - 1, MaxUint32, 0, 1},                         // sum
						{1, 2, 3, 4},                                             // min
						{MaxUint32, MaxUint32, MaxUint32, MaxUint32},             // max
						{MaxUint32, MaxUint32 - 1, MaxUint32 - 2, MaxUint32 - 3}, // prod
						{1, 1, 1, 1}, // land
						{1, 1, 1, 1}, // lor
						{0, 0, 0, 0}, // lxor
						{1, 2, 3, 4}, // band
						{MaxUint32, MaxUint32, MaxUint32, MaxUint32},                 // bor
						{MaxUint32 - 1, MaxUint32 - 2, MaxUint32 - 3, MaxUint32 - 4}, // bxor
					}

					x := make([]uint32, 4)
					setSliceUint32(x, int(A.Rank()), 0)
					res := make([]uint32, len(x))
					err := A.ReduceUint32s(res, x, op, root)
					valid := isValidDataTypeForOp(Uint, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]uint32, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualUint32(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				t.Run("int32", func(t *testing.T) {
					results := [...][]int32{
						{-2, -1, 0, 1},   // sum
						{-1, -1, -1, -1}, // min
						{1, 2, 3, 4},     // max
						{-1, -2, -3, -4}, // prod
						{1, 1, 1, 1},     // land
						{1, 1, 1, 1},     // lor
						{0, 0, 0, 0},     // lxor
						{1, 2, 3, 4},     // band
						{-1, -1, -1, -1}, // bor
						{-2, -3, -4, -5}, // bxor
					}
					x := make([]int32, 4)
					setSliceInt32(x, int(A.Rank()), 0)
					res := make([]int32, len(x))
					err := A.ReduceInt32s(res, x, op, root)
					valid := isValidDataTypeForOp(Int, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]int32, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualInt32(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				t.Run("uint64", func(t *testing.T) {
					results := [...][]uint64{
						{MaxUint64 - 1, MaxUint64, 0, 1},                         // sum
						{1, 2, 3, 4},                                             // min
						{MaxUint64, MaxUint64, MaxUint64, MaxUint64},             // max
						{MaxUint64, MaxUint64 - 1, MaxUint64 - 2, MaxUint64 - 3}, // prod
						{1, 1, 1, 1}, // land
						{1, 1, 1, 1}, // lor
						{0, 0, 0, 0}, // lxor
						{1, 2, 3, 4}, // band
						{MaxUint64, MaxUint64, MaxUint64, MaxUint64},                 // bor
						{MaxUint64 - 1, MaxUint64 - 2, MaxUint64 - 3, MaxUint64 - 4}, // bxor
					}

					x := make([]uint64, 4)
					setSliceUint64(x, int(A.Rank()), 0)
					res := make([]uint64, len(x))
					err := A.ReduceUint64s(res, x, op, root)
					valid := isValidDataTypeForOp(Ulong, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]uint64, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualUint64(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				t.Run("int64", func(t *testing.T) {
					results := [...][]int64{
						{-2, -1, 0, 1},   // sum
						{-1, -1, -1, -1}, // min
						{1, 2, 3, 4},     // max
						{-1, -2, -3, -4}, // prod
						{1, 1, 1, 1},     // land
						{1, 1, 1, 1},     // lor
						{0, 0, 0, 0},     // lxor
						{1, 2, 3, 4},     // band
						{-1, -1, -1, -1}, // bor
						{-2, -3, -4, -5}, // bxor
					}
					x := make([]int64, 4)
					setSliceInt64(x, int(A.Rank()), 0)
					res := make([]int64, len(x))
					err := A.ReduceInt64s(res, x, op, root)
					valid := isValidDataTypeForOp(Long, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]int64, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualInt64(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				t.Run("float32", func(t *testing.T) {
					results := [...][]float32{
						{-2, -1, 0, 1},   // sum
						{-1, -1, -1, -1}, // min
						{1, 2, 3, 4},     // max
						{-1, -2, -3, -4}, // prod
						{}, // land - not tested
						{}, // lor - not tested
						{}, // lxor - not tested
						{}, // band - not tested
						{}, // bor - not tested
						{}, // bxor - not tested
					}
					x := make([]float32, 4)
					setSliceFloat32(x, int(A.Rank()), 0)
					res := make([]float32, len(x))
					err := A.ReduceFloat32s(res, x, op, root)
					valid := isValidDataTypeForOp(Float, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]float32, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualFloat32(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				t.Run("float64", func(t *testing.T) {
					results := [...][]float64{
						{-2, -1, 0, 1},   // sum
						{-1, -1, -1, -1}, // min
						{1, 2, 3, 4},     // max
						{-1, -2, -3, -4}, // prod
						{}, // land - not tested
						{}, // lor - not tested
						{}, // lxor - not tested
						{}, // band - not tested
						{}, // bor - not tested
						{}, // bxor - not tested
					}
					x := make([]float64, 4)
					setSliceFloat64(x, int(A.Rank()), 0)
					res := make([]float64, len(x))
					err := A.ReduceFloat64s(res, x, op, root)
					valid := isValidDataTypeForOp(Double, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]float64, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualFloat64(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				t.Run("complex128", func(t *testing.T) {
					results := [...][]complex128{
						{(-2 - 2.9i), (-1 - 2.8i), (0 - 2.7i), (1 - 2.6i)}, // sum
						{}, // min - not tested
						{}, // max - not tested

						{(2.2 - 1.8i), (4.4 - 3.6i), (6.6 - 5.4i), (8.8 - 7.199999999999999i)}, //prod
						{}, // prod - not tested
						{}, // land - not tested
						{}, // lor - not tested
						{}, // lxor - not tested
						{}, // band - not tested
						{}, // bor - not tested
						{}, // bxor - not tested
					}
					x := make([]complex128, 4)
					setSliceComplex128(x, int(A.Rank()), 0)
					res := make([]complex128, len(x))
					err := A.ReduceComplex128s(res, x, op, root)
					valid := isValidDataTypeForOp(Complex, op)
					if err != nil {
						if valid {
							t.Errorf("Improper error was thrown: valid data type for the op was supplied")
						}
						return
					}
					if !valid {
						t.Errorf("Error should have been thrown: invalid data type for op not properly detected.")
					}
					exp := make([]complex128, 4)
					if A.Rank() == root {
						exp = results[opidx]
					}
					if !chkArraysEqualComplex128(res, exp) {
						t.Errorf("received %v, expected %v", res, exp)
					}
				})
				A.Barrier()

				// t.Run("complex128", func(t *testing.T) {
				// 	x := make([]complex128, 4)
				// 	setSliceComplex128(x, int(A.Rank()), 0)
				// 	res := make([]complex128, len(x))
				// 	A.ReduceSumComplex128s(res, x, root)
				// 	if A.Rank() == root {
				// 		chkArraysEqualComplex128(t, res, []complex128{complex(-2, -2.9), complex(-1, -2.8), complex(0, -2.7), complex(1, -2.6)})
				// 	} else {
				// 		chkArraysEqualComplex128(t, res, []complex128{0, 0, 0, 0})
				// 	}
				// })
				A.Barrier()
			})
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
	// if A.Rank() != 0 {
	// 	os.Stdout, _ = os.Open(os.DevNull)
	// }
	// B := NewCommunicator([]int{0, 1, 2, 3})

	t.Run("Bcast", bcast(A, t))
	A.Barrier()
	t.Run("Reduce", reduce(A, t))
	A.Barrier()
	// t.Run("ReduceSumFloat64s", func(t *testing.T) {
	// 	root := 3
	// 	x := make([]float64, 4)
	// 	setSliceFloat64(x, int(A.Rank()), 0)
	// 	res := make([]float64, len(x))
	// 	A.ReduceSumFloat64s(res, x, root)
	// 	if A.Rank() == root {
	// 		chkArraysEqualFloat64(t, res, []float64{-2, -1, 0, 1})
	// 	} else {
	// 		chkArraysEqualFloat64(t, res, []float64{0, 0, 0, 0})
	// 	}
	// })
	// A.Barrier()
	//
	// // AllReduceSum
	// t.Run("AllReduceSumFloat64s", func(t *testing.T) {
	// 	x := make([]float64, 4)
	// 	res := make([]float64, 4)
	// 	setSliceFloat64(x, int(A.Rank()), 0)
	// 	A.AllReduceSumFloat64s(res, x)
	// 	chkArraysEqualFloat64(t, res, []float64{-2, -1, 0, 1})
	// })
	// A.Barrier()
	//
	// // AllReduceMin
	// t.Run("AllReduceMinFloat64s", func(t *testing.T) {
	// 	x := make([]float64, 4)
	// 	setSliceFloat64(x, int(A.Rank()), -3.5)
	// 	res := make([]float64, len(x))
	// 	A.AllReduceMinFloat64s(res, x)
	// 	// fmt.Println("allreducemin: rank", A.Rank(), "res = ", res)
	// 	// fmt.Println("allreduceminL rank", A.Rank(), "  x = ", x)
	// 	chkArraysEqualFloat64(t, res, []float64{-2.5, -1.5, -1, -1})
	// })
	// A.Barrier()
	//
	// // AllReduceMax
	// t.Run("AllReduceMaxFloat64s", func(t *testing.T) {
	// 	x := make([]float64, 4)
	// 	setSliceFloat64(x, int(A.Rank()), 3.5)
	// 	res := make([]float64, len(x))
	// 	A.AllReduceMaxFloat64s(res, x)
	// 	chkArraysEqualFloat64(t, res, []float64{4.5, 5.5, 6.5, 7.5})
	// })
	// A.Barrier()
	//
	// Send & Recv
	t.Run("SendFloat64s/RecvFloat64s", func(t *testing.T) {
		if A.Rank() == 0 {
			s := []float64{123, 123, 123, 123}
			for k := 1; k <= 3; k++ {
				A.SendFloat64s(s, k, 1)
			}
		} else {
			y := A.RecvFloat64s(0, 1)
			chkArraysEqualFloat64(y, []float64{123, 123, 123, 123})
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
			chkArraysEqualInt64(y, []int64{123, 123, 123, 123})
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
