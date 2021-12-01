// Copyright 2019 Seth Bromberger. All Rights Reserved.

// This code was derived from / inspired by Gosl:
// Copyright 2016 The Gosl Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows
// +build !windows

//go:generate stringer -type=DataType
//go:generate stringer -type=Op

// Package mpi wraps the Message Passing Interface for parallel computations
package mpi

/*
#include "mpi.h"

MPI_Comm     World     = MPI_COMM_WORLD;
MPI_Status*  StIgnore  = MPI_STATUS_IGNORE;

#define DOUBLE_COMPLEX double complex
*/
import "C"

import (
	"fmt"
	"log"
	"runtime"
	"unsafe"
)

type DataType uint8

const (
	MPI_ANY_SOURCE = C.MPI_ANY_SOURCE
	MPI_ANY_TAG    = C.MPI_ANY_TAG
)

const (
	// These constants represent (a subset of) MPI datatypes.
	Byte    DataType = iota
	Uint             // This maps to a uint32 in go.
	Int              // This maps to an int32 in go.
	Ulong            // This maps to a uint64 in go.
	Long             // This maps to an int64 in go.
	Float            // This maps to a float32 in go
	Double           // This maps to a float64 in go.
	Complex          // This maps to a complex128 in go.
)

var dataTypes = [...]C.MPI_Datatype{
	C.MPI_BYTE,
	C.MPI_UINT32_T,
	C.MPI_INT32_T,
	C.MPI_UINT64_T,
	C.MPI_INT64_T,
	C.MPI_FLOAT,
	C.MPI_DOUBLE,
	C.MPI_DOUBLE_COMPLEX,
}

type Op uint8

const (
	OpSum Op = iota
	OpMin
	OpMax
	OpProd
	OpLand
	OpLor
	OpLxor
	OpBand
	OpBor
	OpBxor
)

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

// Returns true if the datatype can be used for the given operation.
// This is needed because boolean/logical operators are invalid for non-ints,
// and complex numbers have no ordering.
func isValidDataTypeForOp(d DataType, o Op) bool {
	if o == OpLand || o == OpLor || o == OpLxor || o == OpBand || o == OpBor || o == OpBxor {
		return d == Byte || d == Uint || d == Int || d == Ulong || d == Long
	}
	if o == OpMin || o == OpMax {
		return d != Complex
	}
	return true
}

// Status wraps an MPI_Status structure.
type Status struct {
	mpiStatus C.MPI_Status
}

func (o Communicator) GetAttr(attribute int) (int, bool, error) {
	var n int
	var found C.int

	x := C.MPI_Comm_get_attr(o.comm, C.int(attribute), unsafe.Pointer(&n), &found)
	if x != C.MPI_SUCCESS {
		return int(n), int(found) == 1, fmt.Errorf("GetAttr returned error %d\n", x)
	}
	return int(n), int(found) == 1, nil
}

func (o Communicator) GetMaxTag() (int, error) {
	x, found, err := o.GetAttr(C.MPI_TAG_UB)
	if !found {
		return -1, fmt.Errorf("No max tag value found")
	}
	if err != nil {
		return -1, err
	}
	return x, nil
}

// Probe issues an MPI Probe and returns a Status structure.
func (o Communicator) Probe(source int, tag int) Status {
	var s Status
	C.MPI_Probe(C.int(source), C.int(tag), o.comm, &(s.mpiStatus))
	return s
}

func (o Communicator) Mprobe(source int, tag int) (Status, C.MPI_Message) {
	var s Status
	var msg C.MPI_Message
	C.MPI_Mprobe(C.int(source), C.int(tag), o.comm, &msg, &(s.mpiStatus))
	return s, msg
}

// GetCount returns a count of elements of type `t` from a Status object.
func (s Status) GetCount(t DataType) int {
	var n C.int
	C.MPI_Get_count(&s.mpiStatus, dataTypes[t], &n)
	return int(n)
}

// GetError returns the error code from a Status object.
func (s Status) GetError() int {
	return int(s.mpiStatus.MPI_ERROR)
}

// GetSource returns the source (sender) of an MPI message.
func (s Status) GetSource() int {
	return int(s.mpiStatus.MPI_SOURCE)
}

// GetTag returns the tag associated with the MPI channel.
func (s Status) GetTag() int {
	return int(s.mpiStatus.MPI_TAG)
}

// IsOn tells whether MPI is on or not
//  NOTE: this returns true even after Stop
func IsOn() bool {
	var flag C.int
	C.MPI_Initialized(&flag)
	return flag != 0
}

// Start initialises MPI
func Start(threaded bool) {
	if threaded {
		var x C.int
		C.MPI_Init_thread(nil, nil, C.MPI_THREAD_MULTIPLE, &x)
		if x != C.MPI_THREAD_MULTIPLE {
			log.Fatalf("Requested threading support %d not available (%d).", C.MPI_THREAD_MULTIPLE, x)
		}
	} else {
		C.MPI_Init(nil, nil)
	}
}

// Stop finalises MPI
func Stop() {
	C.MPI_Finalize()
}

// WorldRank returns the processor rank/ID within the World communicator
func WorldRank() (rank int) {
	var r int32
	C.MPI_Comm_rank(C.World, (*C.int)(unsafe.Pointer(&r)))
	return int(r)
}

// WorldSize returns the number of processors in the World communicator
func WorldSize() (size int) {
	var s int32
	C.MPI_Comm_size(C.World, (*C.int)(unsafe.Pointer(&s)))
	return int(s)
}

func WorldTime() float64 {
	return float64(C.MPI_Wtime())
}

// Communicator holds the World communicator or a subset communicator
type Communicator struct {
	comm  C.MPI_Comm
	group C.MPI_Group
}

// NewCommunicator creates a new communicator or returns the World communicator
//   ranks -- World indices of processors in this Communicator.
//            use nil or empty to get the World Communicator
func NewCommunicator(ranks []int) Communicator {
	var o Communicator
	if len(ranks) == 0 {
		o.comm = C.World
		C.MPI_Comm_group(C.World, &o.group)
		return o
	}
	rs := make([]int32, len(ranks))
	for i := 0; i < len(ranks); i++ {
		rs[i] = int32(ranks[i])
	}
	n := C.int(len(ranks))
	r := (*C.int)(unsafe.Pointer(&rs[0]))
	var wgroup C.MPI_Group
	C.MPI_Comm_group(C.World, &wgroup)
	C.MPI_Group_incl(wgroup, n, r, &o.group)
	C.MPI_Comm_create(C.World, o.group, &o.comm)
	return o
}

// Rank returns the processor rank/ID
func (o Communicator) Rank() (rank int) {
	var r int32
	C.MPI_Comm_rank(o.comm, (*C.int)(unsafe.Pointer(&r)))
	return int(r)
}

// Size returns the number of processors
func (o Communicator) Size() (size int) {
	var s int32
	C.MPI_Comm_size(o.comm, (*C.int)(unsafe.Pointer(&s)))
	return int(s)
}

// Abort aborts MPI
func (o Communicator) Abort(errcode int) {
	C.MPI_Abort(o.comm, C.int(errcode))
}

// Barrier forces synchronisation
func (o Communicator) Barrier() {
	C.MPI_Barrier(o.comm)
}

// BcastBytes broadcasts slice from root `root` to all other processors
func (o Communicator) BcastBytes(x []byte, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Int], C.int(root), o.comm)
}

// BcastUint32s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastUint32s(x []uint32, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Uint], C.int(root), o.comm)
}

// BcastInt32s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastInt32s(x []int32, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Int], C.int(root), o.comm)
}

// BcastUint64s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastUint64s(x []uint64, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Ulong], C.int(root), o.comm)
}

// BcastInt64s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastInt64s(x []int64, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Long], C.int(root), o.comm)
}

// BcastFloat32s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastFloat32s(x []float32, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Float], C.int(root), o.comm)
}

// BcastFloat64s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastFloat64s(x []float64, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Double], C.int(root), o.comm)
}

// BcastComplex128s broadcasts slice from root `root` to all other processors
func (o Communicator) BcastComplex128s(x []complex128, root int) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Complex], C.int(root), o.comm)
}

// ReduceBytes performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceBytes(dest, orig []byte, op Op, root int) error {
	d := Byte
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceUint32s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceUint32s(dest, orig []uint32, op Op, root int) error {
	d := Uint
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceInt32s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceInt32s(dest, orig []int32, op Op, root int) error {
	d := Int
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceUInt64s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceUint64s(dest, orig []uint64, op Op, root int) error {
	d := Ulong
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceInt64s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceInt64s(dest, orig []int64, op Op, root int) error {
	d := Long
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceFloat32s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceFloat32s(dest, orig []float32, op Op, root int) error {
	d := Float
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceFloat64s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceFloat64s(dest, orig []float64, op Op, root int) error {
	d := Double
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// ReduceComplex128s performs a distributed reduce operation on bytes, accumulating the operation on the given root.
// Note: dest and orig must be different slices.
func (o Communicator) ReduceComplex128s(dest, orig []complex128, op Op, root int) error {
	d := Complex
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], C.int(root), o.comm)
	return nil
}

// AllreduceBytes performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceBytes(dest, orig []byte, op Op, root int) error {
	d := Byte
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceUint32s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceUint32s(dest, orig []uint32, op Op, root int) error {
	d := Uint
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceInt32s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceInt32s(dest, orig []int32, op Op, root int) error {
	d := Int
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceUint64s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceUint64s(dest, orig []uint64, op Op, root int) error {
	d := Ulong
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceInt64s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceInt64s(dest, orig []int64, op Op, root int) error {
	d := Long
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceFloat32s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceFloat32s(dest, orig []float32, op Op, root int) error {
	d := Float
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceFloat64s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceFloat64s(dest, orig []float64, op Op, root int) error {
	d := Double
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// AllreduceComplex128s performs a distributed reduce operation on bytes, accumulating the operation on all roots.
// Note: dest and orig must be different slices.
func (o Communicator) AllreduceComplex128s(dest, orig []complex128, op Op, root int) error {
	d := Complex
	if !isValidDataTypeForOp(d, op) {
		return fmt.Errorf("DataType %v cannot be used with Operation %v", d, op)
	}
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[d], ops[op], o.comm)
	return nil
}

// SendBytes sends values to processor toID with given tag
func (o Communicator) SendBytes(vals []byte, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Byte], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocBytes receives values from processor fromId with given tag
func (o Communicator) RecvPreallocBytes(vals []byte, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}

	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Byte], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// MrecvPreallocBytes receives values from processor fromId with given tag with threading
func (o Communicator) MrecvPreallocBytes(vals []byte, fromID int, tag int, msg C.MPI_Message) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}

	C.MPI_Mrecv(buf, C.int(len(vals)), dataTypes[Byte], &msg, &(status.mpiStatus))
	return status
}

// RecvBytes returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvBytes(fromID int, tag int) ([]byte, Status) {
	l := o.Probe(fromID, tag).GetCount(Byte)
	buf := make([]byte, l)
	status := o.RecvPreallocBytes(buf, fromID, tag)
	return buf, status
}

// MrecvBytes returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) MrecvBytes(fromID int, tag int) ([]byte, Status) {
	runtime.LockOSThread()
	pstatus, msg := o.Mprobe(fromID, tag)
	l := pstatus.GetCount(Byte)
	buf := make([]byte, l)
	status := o.MrecvPreallocBytes(buf, fromID, tag, msg)
	return buf, status
}

// SendUint32s sends values to processor toID with given tag
func (o Communicator) SendUInt32s(vals []uint32, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Uint], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocUint32s receives values from processor fromId with given tag
func (o Communicator) RecvPreallocUint32s(vals []uint32, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Uint], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))

	return status
}

// RecvUint32s returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvUint32s(fromID int, tag int) ([]uint32, Status) {
	l := o.Probe(fromID, tag).GetCount(Uint)
	buf := make([]uint32, l)
	status := o.RecvPreallocUint32s(buf, fromID, tag)
	return buf, status
}

// SendInt32s sends values to processor toID with given tag
func (o Communicator) SendInt32s(vals []int32, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Int], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocInt32s receives values from processor fromId with given tag
func (o Communicator) RecvPreallocInt32s(vals []int32, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Int], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// RecvInt32s returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvInt32s(fromID int, tag int) ([]int32, Status) {
	l := o.Probe(fromID, tag).GetCount(Int)
	buf := make([]int32, l)
	status := o.RecvPreallocInt32s(buf, fromID, tag)
	return buf, status
}

// SendUint64s sends values to processor toID with given tag
func (o Communicator) SendUint64s(vals []uint64, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Ulong], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocUint64s receives values from processor fromId with given tag
func (o Communicator) RecvPreallocUint64s(vals []uint64, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Ulong], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// RecvUint64s returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvUint64s(fromID int, tag int) ([]uint64, Status) {
	l := o.Probe(fromID, tag).GetCount(Ulong)
	buf := make([]uint64, l)
	status := o.RecvPreallocUint64s(buf, fromID, tag)
	return buf, status
}

// SendInt64s sends values to processor toID with given tag
func (o Communicator) SendInt64s(vals []int64, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Long], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocInt64s receives values from processor fromId with given tag
func (o Communicator) RecvPreallocInt64s(vals []int64, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Long], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// RecvInt64s returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvInt64s(fromID int, tag int) ([]int64, Status) {
	l := o.Probe(fromID, tag).GetCount(Long)
	buf := make([]int64, l)
	status := o.RecvPreallocInt64s(buf, fromID, tag)
	return buf, status
}

// SendFloat64s sends values to processor toID with given tag
func (o Communicator) SendFloat64s(vals []float64, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Double], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocFloat64s receives values from processor fromId with given tag
func (o Communicator) RecvPreallocFloat64s(vals []float64, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Double], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// RecvFloat64s returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvFloat64s(fromID int, tag int) ([]float64, Status) {
	l := o.Probe(fromID, tag).GetCount(Double)
	buf := make([]float64, l)
	status := o.RecvPreallocFloat64s(buf, fromID, tag)
	return buf, status
}

// SendComplex128s sends values to processor toID with given tag
func (o Communicator) SendComplex128s(vals []complex128, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Complex], C.int(toID), C.int(tag), o.comm)
}

// RecvPreallocComplex128s receives values from processor fromId with given tag
func (o Communicator) RecvPreallocComplex128s(vals []complex128, fromID int, tag int) Status {
	buf := unsafe.Pointer(&vals[0])
	status := Status{}
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Complex], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// RecvComplex128s returns a slice of bytes received from processor fromId with given tag.
func (o Communicator) RecvComplex128s(fromID int, tag int) ([]complex128, Status) {
	l := o.Probe(fromID, tag).GetCount(Complex)
	buf := make([]complex128, l)
	status := o.RecvPreallocComplex128s(buf, fromID, tag)
	return buf, status
}

//////////////////////////////////////////////////////////////////////////////
// SendByte sends one value to processor toID with given tag
func (o Communicator) SendByte(v byte, toID int, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Byte], C.int(toID), C.int(tag), o.comm)
}

// RecvByte receives one value from processor fromId with given tag
func (o Communicator) RecvByte(fromID, tag int) (byte, Status) {
	var v byte
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Byte], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// SendUint sends one value to processor toID with given tag
func (o Communicator) SendUint32(v uint32, toID int, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Uint], C.int(toID), C.int(tag), o.comm)
}

// RecvUint receives one value from processor fromId with given tag
func (o Communicator) RecvUint32(fromID, tag int) (uint32, Status) {
	var v uint32
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Uint], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// SendInt sends one value to processor toID with given tag
func (o Communicator) SendInt32(v int32, toID int, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Int], C.int(toID), C.int(tag), o.comm)
}

// RecvInt receives one value from processor fromId with given tag
func (o Communicator) RecvInt32(fromID, tag int) (int32, Status) {
	var v int32
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Int], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// SendUint32 sends one value to processor toID with given tag
func (o Communicator) SendUint64(v uint64, toID int, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Ulong], C.int(toID), C.int(tag), o.comm)
}

// RecvUlong receives one value from processor fromId with given tag
func (o Communicator) RecvUint64(fromID, tag int) (uint64, Status) {
	var v uint64
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Ulong], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// SendLong sends one value to processor toID with given tag
func (o Communicator) SendInt64(v int64, toID int, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Long], C.int(toID), C.int(tag), o.comm)
}

// RecvLong receives one value from processor fromId with given tag
func (o Communicator) RecvInt64(fromID, tag int) (int64, Status) {
	var v int64
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Long], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// SendDouble sends one value to processor toID with given tag
func (o Communicator) SendFloat64(v float64, toID int, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Double], C.int(toID), C.int(tag), o.comm)
}

// RecvDouble receives one value from processor fromId with given tag
func (o Communicator) RecvFloat64(fromID, tag int) (float64, Status) {
	var v float64
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Double], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// SendComplex128 sends one value to processor toID (integer version)
func (o Communicator) SendComplex128(v complex128, toID, tag int) {
	buf := unsafe.Pointer(&v)
	C.MPI_Send(buf, 1, dataTypes[Complex], C.int(toID), C.int(tag), o.comm)
}

// RecvComplex128 receives one value from processor fromId
func (o Communicator) RecvComplex128(fromID, tag int) (complex128, Status) {
	var v complex128
	buf := unsafe.Pointer(&v)
	status := Status{}
	C.MPI_Recv(buf, 1, dataTypes[Complex], C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))

	return v, status
}

// SendString is a convenience function to send one string to processor toID with given tag.
func (o Communicator) SendString(s string, toID, tag int) {
	o.SendBytes([]byte(s), toID, tag)
}

// RecvString is a convenience function to receive a string from processor fromId with given tag.
func (o Communicator) RecvString(fromID, tag int) (string, Status) {
	recv_bytes, status := o.RecvBytes(fromID, tag)
	return string(recv_bytes), status
}

// IProbe will return a boolean indicating whether a message is
// waiting from a source with a given tag, and a status structure.
func (o Communicator) Iprobe(source, tag int) (bool, Status) {
	var s Status
	var b C.int

	C.MPI_Iprobe(C.int(source), C.int(tag), o.comm, &b, &(s.mpiStatus))
	return b == 1, s
}
