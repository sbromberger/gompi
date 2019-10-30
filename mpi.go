// Copyright 2019 Seth Bromberger. All Rights Reserved.

// This code was derived from / inspired by Gosl:
// Copyright 2016 The Gosl Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

// Package mpi wraps the Message Passing Interface for parallel computations
package mpi

/*
#include "mpi.h"

MPI_Comm     World     = MPI_COMM_WORLD;
MPI_Op       OpSum     = MPI_SUM;
MPI_Op       OpMin     = MPI_MIN;
MPI_Op       OpMax     = MPI_MAX;
MPI_Status*  StIgnore  = MPI_STATUS_IGNORE;

#define DOUBLE_COMPLEX double complex
*/
import "C"

import (
	"unsafe"
)

type DataType uint8

const (
	// These constants index MPI datatypes in the following map.
	Byte DataType = iota
	Unsigned
	Int
	Long
	Double
	Complex
)

var dataTypes = [...]C.MPI_Datatype{C.MPI_BYTE, C.MPI_UNSIGNED, C.MPI_INT, C.MPI_LONG, C.MPI_DOUBLE, C.MPI_DOUBLE_COMPLEX}

// Status envelops an MPI_Status structure.
type Status struct {
	mpiStatus C.MPI_Status
}

// Probe issues an MPI Probe and returns a Status structure.
func (o Communicator) Probe(source int, tag int) Status {
	var s Status
	C.MPI_Probe(C.int(source), C.int(tag), o.comm, &(s.mpiStatus))
	return s
}

// GetCountI returns a count of 64-bit integers from a status object.
func (s Status) GetCount(t DataType) int {
	var n C.int
	C.MPI_Get_count(&s.mpiStatus, dataTypes[t], &n)
	return int(n)
}

// func (s Status) GetCount() int {
// 	var n C.int
// 	C.MPI_Get_count(&s.mpiStatus, C.MPI_BYTE, &n)
// 	// fmt.Println("received ", n, "bytes via probe")
// 	return int(n)
// }
//
// IsOn tells whether MPI is on or not
//  NOTE: this returns true even after Stop
func IsOn() bool {
	var flag C.int
	C.MPI_Initialized(&flag)
	return flag != 0

}

// Start initialises MPI
func Start() {
	C.MPI_Init(nil, nil)
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
func (o Communicator) Abort() {
	C.MPI_Abort(o.comm, 0)
}

// Barrier forces synchronisation
func (o Communicator) Barrier() {
	C.MPI_Barrier(o.comm)
}

// BcastFromRoot broadcasts slice from root (Rank == 0) to all other processors
func (o Communicator) BcastFromRoot(x []float64) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Double], 0, o.comm)
}

// BcastFromRootC broadcasts slice from root (Rank == 0) to all other processors (complex version)
func (o Communicator) BcastFromRootC(x []complex128) {
	buf := unsafe.Pointer(&x[0])
	C.MPI_Bcast(buf, C.int(len(x)), dataTypes[Complex], 0, o.comm)
}

// ReduceSum sums all values in 'orig' to 'dest' in root (Rank == 0) processor
//   NOTE (important): orig and dest must be different slices
func (o Communicator) ReduceSum(dest, orig []float64) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Double], C.OpSum, 0, o.comm)
}

// ReduceSumC sums all values in 'orig' to 'dest' in root (Rank == 0) processor (complex version)
//   NOTE (important): orig and dest must be different slices
func (o Communicator) ReduceSumC(dest, orig []complex128) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Reduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Complex], C.OpSum, 0, o.comm)
}

// AllReduceSum combines all values from orig into dest summing values
//   NOTE (important): orig and dest must be different slices
func (o Communicator) AllReduceSum(dest, orig []float64) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Double], C.OpSum, o.comm)
}

// AllReduceSumC combines all values from orig into dest summing values (complex version)
//   NOTE (important): orig and dest must be different slices
func (o Communicator) AllReduceSumC(dest, orig []complex128) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Complex], C.OpSum, o.comm)
}

// AllReduceMin combines all values from orig into dest picking minimum values
//   NOTE (important): orig and dest must be different slices
func (o Communicator) AllReduceMin(dest, orig []float64) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Double], C.OpMin, o.comm)
}

// AllReduceMax combines all values from orig into dest picking minimum values
//   NOTE (important): orig and dest must be different slices
func (o Communicator) AllReduceMax(dest, orig []float64) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Double], C.OpMax, o.comm)
}

// AllReduceMinI combines all values from orig into dest picking minimum values (integer version)
//   NOTE (important): orig and dest must be different slices
func (o Communicator) AllReduceMinI(dest, orig []int) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Long], C.OpMin, o.comm)
}

// AllReduceMaxI combines all values from orig into dest picking minimum values (integer version)
//   NOTE (important): orig and dest must be different slices
func (o Communicator) AllReduceMaxI(dest, orig []int) {
	sendbuf := unsafe.Pointer(&orig[0])
	recvbuf := unsafe.Pointer(&dest[0])
	C.MPI_Allreduce(sendbuf, recvbuf, C.int(len(dest)), dataTypes[Long], C.OpMax, o.comm)
}

// Send sends values to processor toID
func (o Communicator) Send(vals []float64, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Double], C.int(toID), C.int(tag), o.comm)
}

// Recv receives values from processor fromId
func (o Communicator) Recv(vals []float64, fromID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Double], C.int(fromID), C.int(tag), o.comm, C.StIgnore)
}

// func (o Communicator) RecvWithStatus(fromID int)

// SendC sends values to processor toID (complex version)
func (o Communicator) SendC(vals []complex128, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Complex], C.int(toID), C.int(tag), o.comm)
}

// RecvC receives values from processor fromId (complex version)
func (o Communicator) RecvC(vals []complex128, fromID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Complex], C.int(fromID), C.int(tag), o.comm, C.StIgnore)
}

// SendI sends values to processor toID (integer version)
func (o Communicator) SendI(vals []int, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Long], C.int(toID), C.int(tag), o.comm)
}

// RecvI receives values from processor fromId (integer version)
func (o Communicator) RecvI(vals []int, fromID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Long], C.int(fromID), C.int(tag), o.comm, C.StIgnore)
}

// SendB sends values to processor toID (byte version)
func (o Communicator) SendB(vals []byte, toID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, C.int(len(vals)), dataTypes[Byte], C.int(toID), C.int(tag), o.comm)
}

// RecvB receives values from processor fromId (byte version)
func (o Communicator) RecvB(vals []byte, fromID int, tag int) {
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Recv(buf, C.int(len(vals)), dataTypes[Byte], C.int(fromID), C.int(tag), o.comm, C.StIgnore)
}

// SendOne sends one value to processor toID
func (o Communicator) SendOne(val float64, toID int, tag int) {
	vals := []float64{val}
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, 1, dataTypes[Double], C.int(toID), C.int(tag), o.comm)
}

// RecvOne receives one value from processor fromId
func (o Communicator) RecvOne(fromID, tag int) (val float64) {
	vals := []float64{0}
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Recv(buf, 1, dataTypes[Double], C.int(fromID), C.int(tag), o.comm, C.StIgnore)
	return vals[0]
}

// SendOneString is a convenience function to send one string.
func (o Communicator) SendOneString(s string, toID, tag int) {
	o.SendB([]byte(s), toID, tag)
}

// RecvOneString is a convenience function to receive a string.
func (o Communicator) RecvOneString(fromID, tag int) string {
	l := o.Probe(fromID, tag).GetCount(Byte)
	b := make([]byte, l)
	o.RecvB(b, fromID, tag)
	return string(b)
}

// SendOneI sends one value to processor toID (integer version)
// This sends an int as 64 bits.
func (o Communicator) SendOneI(val int, toID, tag int) {
	vals := []int{val}
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Send(buf, 1, dataTypes[Long], C.int(toID), C.int(tag), o.comm)
}

// RecvOneI receives one value from processor fromId (integer version)
func (o Communicator) RecvOneI(fromID, tag int) (val int) {
	vals := []int{0}
	buf := unsafe.Pointer(&vals[0])
	C.MPI_Recv(buf, 1, dataTypes[Long], C.int(fromID), C.int(tag), o.comm, C.StIgnore)
	return vals[0]
}
