// Copyright 2026 Seth Bromberger. All Rights Reserved.

// This code was derived from / inspired by Gosl:
// Copyright 2016 The Gosl Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows

// Package mpi provides a Go wrapper around the Message Passing Interface (MPI)
// for distributed parallel computation. It supports point-to-point messaging,
// collective operations (broadcast, reduce, allreduce), and communicator
// management.
//
// TODO: once Go supports generic methods, replace the type-specific Bcast*, Reduce*,
// Allreduce*, Send*, Recv*, etc. families with generic methods on *Communicator.
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
	"unsafe"
)

// DataType identifies the MPI datatype corresponding to a Go type.
type DataType uint8

// AnySource and AnyTag are wildcard values for use in receive operations.
const (
	AnySource = C.MPI_ANY_SOURCE
	AnyTag    = C.MPI_ANY_TAG
)

// CommTypeShared is the MPI communicator type for processes sharing memory.
const (
	CommTypeShared = C.MPI_COMM_TYPE_SHARED
)

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

// Status holds the result of a completed MPI operation, including the source,
// tag, error code, and element count of the received message.
type Status struct {
	mpiStatus C.MPI_Status
}

// GetAttr retrieves a communicator attribute by key. It returns the attribute
// value, a boolean indicating whether the attribute was set, and any error.
func (o *Communicator) GetAttr(attribute int) (int, bool, error) {
	var n int
	var found C.int

	x := C.MPI_Comm_get_attr(o.comm, C.int(attribute), unsafe.Pointer(&n), &found)
	if x != C.MPI_SUCCESS {
		return int(n), int(found) == 1, fmt.Errorf("GetAttr returned error %d", x)
	}
	return int(n), int(found) == 1, nil
}

// GetMaxTag returns the maximum tag value supported by this communicator.
func (o *Communicator) GetMaxTag() (int, error) {
	x, found, err := o.GetAttr(C.MPI_TAG_UB)
	if !found {
		return -1, fmt.Errorf("no max tag value found")
	}
	if err != nil {
		return -1, err
	}
	return x, nil
}

// Probe blocks until a message matching source and tag is available, and
// returns its status. The message is not consumed; use a Recv to read it.
func (o *Communicator) Probe(source int, tag int) Status {
	var s Status
	C.MPI_Probe(C.int(source), C.int(tag), o.comm, &(s.mpiStatus))
	return s
}

// Mprobe blocks until a message matching source and tag is available, claims
// it atomically, and returns its status and a message handle. The claimed
// message must be received with MrecvPreallocBytes or MrecvBytes. This is the
// thread-safe alternative to Probe.
func (o *Communicator) Mprobe(source int, tag int) (Status, C.MPI_Message) {
	var s Status
	var msg C.MPI_Message
	C.MPI_Mprobe(C.int(source), C.int(tag), o.comm, &msg, &(s.mpiStatus))
	return s, msg
}

// GetCount returns the number of elements of type t in the received message
// described by this Status.
func (s Status) GetCount[T goTypes]() int {
	var n C.int
	C.MPI_Get_count(&s.mpiStatus, dataTypeOf[T](), &n)
	return int(n)
}

// GetError returns the error code associated with this Status.
func (s Status) GetError() int {
	return int(s.mpiStatus.MPI_ERROR)
}

// GetSource returns the rank of the processor that sent the message described
// by this Status.
func (s Status) GetSource() int {
	return int(s.mpiStatus.MPI_SOURCE)
}

// GetTag returns the tag of the message described by this Status.
func (s Status) GetTag() int {
	return int(s.mpiStatus.MPI_TAG)
}

// IsOn reports whether MPI has been initialised and not yet finalised.
func IsOn() bool {
	var init, fin C.int
	C.MPI_Initialized(&init)
	C.MPI_Finalized(&fin)
	return init != 0 && fin == 0
}

// MPI is a token representing an active MPI session. It is obtained by calling
// Start or StartThreaded and must be used to access world-level operations and
// create communicators. Only one MPI session may exist per process.
type MPI struct{}

// Start initialises MPI and returns a session token. It returns an error if
// MPI is already initialised. MPI's default error handler
// (MPI_ERRORS_ARE_FATAL) will abort the process on any subsequent MPI failure.
func Start() (*MPI, error) {
	if IsOn() {
		return nil, fmt.Errorf("MPI is already initialized")
	}
	C.MPI_Init(nil, nil)
	return &MPI{}, nil
}

// StartThreaded initialises MPI with full thread support (MPI_THREAD_MULTIPLE)
// and returns a session token. It returns an error if MPI is already
// initialised or if the requested threading level is not available.
func StartThreaded() (*MPI, error) {
	if IsOn() {
		return nil, fmt.Errorf("MPI is already initialized")
	}
	var x C.int
	C.MPI_Init_thread(nil, nil, C.MPI_THREAD_MULTIPLE, &x)
	if x != C.MPI_THREAD_MULTIPLE {
		return nil, fmt.Errorf("MPI thread support %d unavailable (got %d).", C.MPI_THREAD_MULTIPLE, x)
	}
	return &MPI{}, nil
}

// Stop finalises MPI. No MPI calls may be made after Stop returns.
func (m *MPI) Stop() {
	C.MPI_Finalize()
}

// WorldRank returns the rank of this process within the world communicator.
func (m *MPI) WorldRank() int {
	var r int32
	C.MPI_Comm_rank(C.World, (*C.int)(unsafe.Pointer(&r)))
	return int(r)
}

// WorldSize returns the number of processes in the world communicator.
func (m *MPI) WorldSize() int {
	var s int32
	C.MPI_Comm_size(C.World, (*C.int)(unsafe.Pointer(&s)))
	return int(s)
}

// WorldTime returns the elapsed wall-clock time in seconds, as reported by
// MPI_Wtime. Useful for portable high-resolution timing.
func (m *MPI) WorldTime() float64 {
	return float64(C.MPI_Wtime())
}

// Communicator wraps an MPI communicator and its associated process group.
// Use NewCommunicator to obtain one.
type Communicator struct {
	comm   C.MPI_Comm
	group  C.MPI_Group
	MaxTag int
}

// NewCommunicator creates a communicator containing the processes identified
// by ranks. If ranks is nil or empty, the world communicator is returned.
func (m *MPI) NewCommunicator(ranks []int) *Communicator {
	var o Communicator
	if len(ranks) == 0 {
		o.comm = C.World
		C.MPI_Comm_group(C.World, &o.group)
		maxtag, err := o.GetMaxTag()
		if err != nil {
			panic(err)
		}
		o.MaxTag = maxtag
		return &o
	}
	rs := make([]int32, len(ranks))
	for i := 0; i < len(ranks); i++ {
		rs[i] = int32(ranks[i])
	}
	n := C.int(len(ranks))
	r := (*C.int)(unsafe.Pointer(unsafe.SliceData(rs)))
	var wgroup C.MPI_Group
	C.MPI_Comm_group(C.World, &wgroup)
	C.MPI_Group_incl(wgroup, n, r, &o.group)
	C.MPI_Comm_create(C.World, o.group, &o.comm)
	return &o
}

// SplitType splits the communicator using MPI_Comm_split_type.
// func (o *Communicator) SplitType(type int)

// Rank returns the rank of this process within the communicator.
func (o *Communicator) Rank() int {
	var r int32
	C.MPI_Comm_rank(o.comm, (*C.int)(unsafe.Pointer(&r)))
	return int(r)
}

// Size returns the number of processes in the communicator.
func (o *Communicator) Size() int {
	var s int32
	C.MPI_Comm_size(o.comm, (*C.int)(unsafe.Pointer(&s)))
	return int(s)
}

// Abort terminates all processes in the communicator with the given error code.
func (o *Communicator) Abort(errcode int) {
	C.MPI_Abort(o.comm, C.int(errcode))
}

// Barrier blocks until all processes in the communicator have called Barrier.
func (o *Communicator) Barrier() {
	C.MPI_Barrier(o.comm)
}

// Bcast broadcasts x from the root process to all other processes in the
// communicator. All processes must call Bcast with the same root and a
// slice of the same length.
func (o *Communicator) Bcast[T goTypes](x []T, root int) {
	C.MPI_Bcast(unsafe.Pointer(unsafe.SliceData(x)), C.int(len(x)), dataTypeOf[T](), C.int(root), o.comm)
}

// Reduce applies op to orig across all processes and writes the result
// into dest on the root process. dest and orig must be different slices.
// Returns an error if op is not valid for the data type.
func (o *Communicator) Reduce[T goTypes](dest, orig []T, op Op, root int) error {
	if !isValidDataTypeForOp[T](op) {
		return fmt.Errorf("DataType %T cannot be used with Operation %v", *new(T), op)
	}
	C.MPI_Reduce(unsafe.Pointer(unsafe.SliceData(orig)), unsafe.Pointer(unsafe.SliceData(dest)), C.int(len(dest)), dataTypeOf[T](), ops[op], C.int(root), o.comm)
	return nil
}

// AllreduceBytes applies op to orig across all processes and writes the result
// into dest on every process. dest and orig must be different slices.
// Returns an error if op is not valid for bytes.
func (o *Communicator) Allreduce[T goTypes](dest, orig []T, op Op, root int) error {
	if !isValidDataTypeForOp[T](op) {
		return fmt.Errorf("DataType %T cannot be used with Operation %v", *new(T), op)
	}
	C.MPI_Allreduce(unsafe.Pointer(unsafe.SliceData(orig)), unsafe.Pointer(unsafe.SliceData(dest)), C.int(len(dest)), dataTypeOf[T](), ops[op], o.comm)
	return nil
}

// Send sends vals to processor toID with the given tag.
func (o *Communicator) Send[T goTypes](vals []T, toID int, tag int) {
	C.MPI_Send(unsafe.Pointer(unsafe.SliceData(vals)), C.int(len(vals)), dataTypeOf[T](), C.int(toID), C.int(tag), o.comm)
}

// RecvPrealloc receives into the preallocated slice vals from processor
// fromID with the given tag, and returns the resulting Status.
func (o *Communicator) RecvPrealloc[T goTypes](vals []T, fromID int, tag int) Status {
	status := Status{}
	C.MPI_Recv(unsafe.Pointer(unsafe.SliceData(vals)), C.int(len(vals)), dataTypeOf[T](), C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return status
}

// MrecvPreallocBytes receives into the preallocated slice vals using the
// matched message handle msg obtained from Mprobe.
func (o *Communicator) MrecvPrealloc[T goTypes](vals []T, msg C.MPI_Message) Status {
	status := Status{}
	C.MPI_Mrecv(unsafe.Pointer(unsafe.SliceData(vals)), C.int(len(vals)), dataTypeOf[T](), &msg, &(status.mpiStatus))
	return status
}

// MrecvBytes receives a byte slice via a matched receive from processor fromID
// with the given tag. It calls Mprobe to atomically claim the message before
// receiving, making it safe for use in multi-threaded programs.
func (o *Communicator) Mrecv[T goTypes](fromID int, tag int) ([]T, Status) {
	pstatus, msg := o.Mprobe(fromID, tag)
	l := pstatus.GetCount[T]()
	buf := make([]T, l)
	status := o.MrecvPrealloc[T](buf, msg)
	return buf, status
}

// Recv allocates and returns a slice received from processor fromID
// with the given tag.
func (o *Communicator) Recv[T goTypes](fromID int, tag int) ([]T, Status) {
	l := o.Probe(fromID, tag).GetCount[T]()
	buf := make([]T, l)
	status := o.RecvPrealloc[T](buf, fromID, tag)
	return buf, status
}

// ////////////////////////////////////////////////////////////////////////////

// SendByte sends a single value to processor toID with the given tag.
func (o *Communicator) SendOne[T goTypes](v T, toID int, tag int) {
	C.MPI_Send(unsafe.Pointer(&v), 1, dataTypeOf[T](), C.int(toID), C.int(tag), o.comm)
}

// RecvOne receives a single value from processor fromID with the given tag.
func (o *Communicator) RecvOne[T goTypes](fromID, tag int) (T, Status) {
	var v T
	status := Status{}
	C.MPI_Recv(unsafe.Pointer(&v), 1, dataTypeOf[T](), C.int(fromID), C.int(tag), o.comm, &(status.mpiStatus))
	return v, status
}

// // SendString sends s to processor toID with the given tag. The string's backing
// // array is aliased directly to avoid a copy; this is safe because MPI_Send is a
// // blocking call that does not retain the pointer beyond its return.
// func (o *Communicator) SendString(s string, toID, tag int) {
// 	buf := unsafe.Slice(unsafe.StringData(s), len(s))
// 	o.SendBytes(buf, toID, tag)
// }

// // RecvString receives a string from processor fromID with the given tag. The
// // returned string aliases the receive buffer directly to avoid a copy; see
// // RecvBytes for constraints on the underlying memory.
// func (o *Communicator) RecvString(fromID, tag int) (string, Status) {
// 	recv_bytes, status := o.RecvBytes(fromID, tag)
// 	return unsafe.String(unsafe.SliceData(recv_bytes), len(recv_bytes)), status
// }

// Iprobe reports whether a message from source with the given tag is available
// without blocking. It returns true and the message Status if a message is
// waiting, or false and a zero Status if not.
func (o *Communicator) Iprobe(source, tag int) (bool, Status) {
	var s Status
	var b C.int

	C.MPI_Iprobe(C.int(source), C.int(tag), o.comm, &b, &(s.mpiStatus))
	return b == 1, s
}
