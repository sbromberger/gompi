# Migration Guide

## Migrating from v0.2 to v0.3

### Initialization and teardown

The package-level `Start`, `Stop`, `WorldRank`, `WorldSize`, and `WorldTime`
functions have been replaced by a session object.

```go
// v0.2
mpi.Start(false)
defer mpi.Stop()
rank := mpi.WorldRank()
size := mpi.WorldSize()
```

```go
// v0.3
m, err := mpi.Start()
if err != nil {
    log.Fatal(err)
}
defer m.Stop()
rank := m.WorldRank()
size := m.WorldSize()
```

For threaded initialization:

```go
// v0.2
mpi.Start(true)

// v0.3
m, err := mpi.StartThreaded()
```

### Initialization check

```go
// v0.2
mpi.IsOn()

// v0.3
mpi.IsInitialized()
```

### Communicators

`NewCommunicator` is now a method on `*MPI` rather than a package-level function.

```go
// v0.2
comm := mpi.NewCommunicator(nil)

// v0.3
comm := m.NewCommunicator(nil)
```

### Send and receive

All type-specific send/receive methods (`SendBytes`, `SendInt32s`, `SendFloat64s`,
etc.) have been replaced by generic methods.

```go
// v0.2
comm.SendFloat64s(vals, toID, tag)
vals, status := comm.RecvFloat64s(fromID, tag)
comm.RecvPreallocFloat64s(vals, fromID, tag)
```

```go
// v0.3
comm.Send(vals, toID, tag)
vals, status := comm.Recv[float64](fromID, tag)
comm.RecvPrealloc(vals, fromID, tag)
```

The same pattern applies to all types: `byte`, `int8`, `int16`, `uint16`,
`int32`, `uint32`, `int64`, `uint64`, `float32`, `float64`, `complex64`,
`complex128`.

### Single-value send/receive

```go
// v0.2
comm.SendFloat64(v, toID, tag)
v, status := comm.RecvFloat64(fromID, tag)
```

```go
// v0.3
comm.SendOne(v, toID, tag)
v, status := comm.RecvOne[float64](fromID, tag)
```

### Broadcast

```go
// v0.2
comm.BcastFloat64s(vals, root)

// v0.3
comm.Bcast(vals, root)
```

### Reduce and Allreduce

```go
// v0.2
comm.ReduceFloat64s(dest, orig, mpi.OpSum, root)
comm.AllreduceFloat64s(dest, orig, mpi.OpSum, root)

// v0.3
comm.Reduce(dest, orig, mpi.OpSum, root)
comm.Allreduce(dest, orig, mpi.OpSum)
```

Note that `Allreduce` no longer takes a `root` parameter. The v0.2 parameter
was accepted but never used; v0.3 removes it.

### Status

`Status` is now a value type. Methods that previously returned `*Status` now
return `Status`.

`Count` (formerly `GetCount`) is now generic and no longer takes a `DataType` argument, and has
been renamed along with other `Status` methods.

| v0.2 | v0.3 |
|------|------|
| `Status.GetSource()` | `Status.Source()` |
| `Status.GetTag()` | `Status.Tag()` |
| `Status.GetError()` | `Status.Error()` |
| `Status.GetCount(mpi.Byte)` | `Status.Count[byte]()` |

### Mprobe and Mrecv

The v0.2 API exposed `C.MPI_Message` directly. In v0.3 this is encapsulated
in `MatchedMessage`.

```go
// v0.2
status, msg := comm.Mprobe(fromID, tag)
n := status.GetCount(mpi.Byte)
buf := make([]byte, n)
comm.MrecvPreallocBytes(buf, fromID, tag, msg)
```

```go
// v0.3
m := comm.Mprobe(fromID, tag)
buf, status := m.Recv[byte]()
// or, with a preallocated buffer:
status = m.RecvPrealloc(buf)
```

`Mrecv` is also available as a single call:

```go
buf, status := comm.Mrecv[byte](fromID, tag)
```

`MatchedMessage` methods follow the same naming convention as `Status`:

| v0.2 | v0.3 |
|------|------|
| `MatchedMessage.GetSource()` | `MatchedMessage.Source()` |
| `MatchedMessage.GetTag()` | `MatchedMessage.Tag()` |
| `MatchedMessage.GetError()` | `MatchedMessage.Error()` |
| `MatchedMessage.GetCount[T]()` | `MatchedMessage.Count[T]()` |

### Communicator methods

| v0.2 | v0.3 |
|------|------|
| `Communicator.GetAttr()` | `Communicator.Attr()` |
| `Communicator.GetMaxTag()` | `Communicator.MaxTag()` |

### Communicator.MaxTag field removed

The exported `MaxTag` field on `Communicator` has been unexported. Use
`Communicator.MaxTag()` if you need this value.

### DataType removed

The exported `DataType` constants (`mpi.Byte`, `mpi.Int`, `mpi.Float`,
`mpi.Double`, etc.) have been removed. Type information is now conveyed through
Go generics and is not part of the public API.

### Iprobe

`Iprobe` now returns a value `Status` rather than a pointer.

### String convenience methods

`SendString` and `RecvString` have been removed. Use `Send` and `Recv` with a
`[]byte` conversion:

```go
// v0.2
comm.SendString(s, toID, tag)
s, status := comm.RecvString(fromID, tag)

// v0.3
comm.Send([]byte(s), toID, tag)
b, status := comm.Recv[byte](fromID, tag)
s := string(b)
```
