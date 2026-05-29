//go:build !windows

package mpi

/*
#include "mpi.h"

#define DOUBLE_COMPLEX double complex
*/
import "C"

//go:generate stringer -type=Op -output=enum_string.go

func dataTypeOf[T goTypes]() C.MPI_Datatype {
	var zero T
	switch any(zero).(type) {
	case byte:
		return C.MPI_BYTE
	case uint32:
		return C.MPI_UINT32_T
	case int32:
		return C.MPI_INT32_T
	case uint64:
		return C.MPI_UINT64_T
	case int64:
		return C.MPI_INT64_T
	case float32:
		return C.MPI_FLOAT
	case float64:
		return C.MPI_DOUBLE
	case complex128:
		return C.MPI_DOUBLE_COMPLEX
	default:
		panic("unreachable")
	}
}

// DataType constants identify the supported MPI datatypes and their Go equivalents.
// const (
// 	Byte    DataType = iota // MPI_BYTE    → byte
// 	Uint                    // MPI_UINT32_T → uint32
// 	Int                     // MPI_INT32_T  → int32
// 	Ulong                   // MPI_UINT64_T → uint64
// 	Long                    // MPI_INT64_T  → int64
// 	Float                   // MPI_FLOAT    → float32
// 	Double                  // MPI_DOUBLE   → float64
// 	Complex                 // MPI_DOUBLE_COMPLEX → complex128
// )

type goTypes interface {
	byte | uint32 | int32 | uint64 | int64 | float32 | float64 | complex128
}

// var dataTypes = [...]C.MPI_Datatype{
// 	C.MPI_BYTE,
// 	C.MPI_UINT32_T,
// 	C.MPI_INT32_T,
// 	C.MPI_UINT64_T,
// 	C.MPI_INT64_T,
// 	C.MPI_FLOAT,
// 	C.MPI_DOUBLE,
// 	C.MPI_DOUBLE_COMPLEX,
// }

// isValidDataTypeForOp reports whether op is valid for the given datatype.
// Logical and bitwise operators require integer types; Min and Max require
// non-complex types.
func isValidDataTypeForOp[T goTypes](o Op) bool {
	var zero T
	switch any(zero).(type) {
	case float32, float64:
		// no logical or bitwise ops
		return o != OpLand && o != OpLor && o != OpLxor &&
			o != OpBand && o != OpBor && o != OpBxor
	case complex128:
		// only sum and product
		return o == OpSum || o == OpProd
	default: // integer types: all ops valid
		return true
	}
}
