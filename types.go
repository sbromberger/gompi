//go:build !windows

package mpi

/*
#include "mpi.h"

#define DOUBLE_COMPLEX double complex
*/
import "C"


type dataType uint8

const (
	dtInt8         dataType = iota // MPI_INT8_T          → int8
	dtByte                         // MPI_BYTE            → byte (uint8)
	dtInt16                        // MPI_INT16_T         → int16
	dtUint16                       // MPI_UINT16_T        → uint16
	dtInt                          // MPI_INT32_T         → int32
	dtUint                         // MPI_UINT32_T        → uint32
	dtLong                         // MPI_INT64_T         → int64
	dtUlong                        // MPI_UINT64_T        → uint64
	dtFloat                        // MPI_FLOAT           → float32
	dtDouble                       // MPI_DOUBLE          → float64
	dtFloatComplex                 // MPI_C_FLOAT_COMPLEX → complex64
	dtComplex                      // MPI_DOUBLE_COMPLEX  → complex128
)

// mpiDataTypes maps each dataType constant to its corresponding C.MPI_Datatype.
//
// MPI type handles are opaque C global variables, not compile-time constants,
// so they cannot be embedded directly in Go const expressions. Any function
// that references them (e.g. via a type switch returning C.MPI_BYTE) incurs
// a cgo call cost that drives the Go inliner budget far above its threshold
// of 80 nodes, preventing inlining entirely.
//
// By loading the handles once into this array at init time, all subsequent
// lookups are plain indexed memory reads with no cgo overhead, and the
// dispatch functions (goDataType, dataTypeOf) remain inlinable.
var mpiDataTypes = [...]C.MPI_Datatype{
	C.MPI_INT8_T,
	C.MPI_BYTE,
	C.MPI_INT16_T,
	C.MPI_UINT16_T,
	C.MPI_INT32_T,
	C.MPI_UINT32_T,
	C.MPI_INT64_T,
	C.MPI_UINT64_T,
	C.MPI_FLOAT,
	C.MPI_DOUBLE,
	C.MPI_C_FLOAT_COMPLEX,
	C.MPI_DOUBLE_COMPLEX,
}

type goTypes interface {
	int8 | byte | int16 | uint16 | int32 | uint32 | int64 | uint64 | float32 | float64 | complex64 | complex128
}

// goDataType returns the dataType index for T. It contains no cgo calls and
// is inlinable (verified via -gcflags="-m=2"; inliner cost ~36, budget 80).
// Returning a plain integer here rather than a C.MPI_Datatype is what keeps
// the cost low: see the mpiDataTypes comment for the full rationale.
func goDataType[T goTypes]() dataType {
	var zero T
	switch any(zero).(type) {
	case int8:
		return dtInt8
	case byte:
		return dtByte
	case int16:
		return dtInt16
	case uint16:
		return dtUint16
	case int32:
		return dtInt
	case uint32:
		return dtUint
	case int64:
		return dtLong
	case uint64:
		return dtUlong
	case float32:
		return dtFloat
	case float64:
		return dtDouble
	case complex64:
		return dtFloatComplex
	case complex128:
		return dtComplex
	default:
		panic("unreachable")
	}
}

// dataTypeOf returns the C.MPI_Datatype for T via a single array lookup.
// Inlinable at cost ~45.
func dataTypeOf[T goTypes]() C.MPI_Datatype {
	return mpiDataTypes[goDataType[T]()]
}

// isValidForOp reports whether op is valid for the given dataType.
func isValidForOp(dt dataType, o Op) bool {
	switch dt {
	case dtFloat, dtDouble:
		return o != OpLand && o != OpLor && o != OpLxor &&
			o != OpBand && o != OpBor && o != OpBxor
	case dtFloatComplex, dtComplex:
		return o == OpSum || o == OpProd
	default:
		return true
	}
}

// isValidDataTypeForOp reports whether op is valid for the Go type T.
func isValidDataTypeForOp[T goTypes](o Op) bool {
	return isValidForOp(goDataType[T](), o)
}

// getDataTypeAndValidate returns the C.MPI_Datatype for T and whether op is
// valid for that type. It calls goDataType once, avoiding the double type
// dispatch that separate dataTypeOf + isValidDataTypeForOp calls would incur.
func getDataTypeAndValidate[T goTypes](o Op) (C.MPI_Datatype, bool) {
	dt := goDataType[T]()
	return mpiDataTypes[dt], isValidForOp(dt, o)
}
