//go:build !windows

package mpi

/*
#include "mpi.h"
*/
import "C"

import "strconv"

// Op identifies an MPI reduction operation.
type Op uint8

// Op constants identify the supported MPI reduction operations.
const (
	OpSum  Op = iota // MPI_SUM
	OpMin            // MPI_MIN
	OpMax            // MPI_MAX
	OpProd           // MPI_PROD
	OpLand           // MPI_LAND (logical and)
	OpLor            // MPI_LOR  (logical or)
	OpLxor           // MPI_LXOR (logical xor)
	OpBand           // MPI_BAND (bitwise and)
	OpBor            // MPI_BOR  (bitwise or)
	OpBxor           // MPI_BXOR (bitwise xor)
)

func (o Op) String() string {
	switch o {
	case OpSum:
		return "OpSum"
	case OpMin:
		return "OpMin"
	case OpMax:
		return "OpMax"
	case OpProd:
		return "OpProd"
	case OpLand:
		return "OpLand"
	case OpLor:
		return "OpLor"
	case OpLxor:
		return "OpLxor"
	case OpBand:
		return "OpBand"
	case OpBor:
		return "OpBor"
	case OpBxor:
		return "OpBxor"
	default:
		return "Op(" + strconv.Itoa(int(o)) + ")"
	}
}

var ops = [...]C.MPI_Op{
	C.MPI_SUM,
	C.MPI_MIN,
	C.MPI_MAX,
	C.MPI_PROD,
	C.MPI_LAND,
	C.MPI_LOR,
	C.MPI_LXOR,
	C.MPI_BAND,
	C.MPI_BOR,
	C.MPI_BXOR,
}
