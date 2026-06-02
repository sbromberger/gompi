#!/usr/bin/env bash
set -euo pipefail

covdir=$(mktemp -d)
testbin=$(mktemp --suffix=.test)
covout=$(mktemp --suffix=.out)
trap 'rm -rf "$covdir" "$testbin" "$covout"' EXIT

go test -c -cover -o "$testbin" .

mpirun -np 4 sh -c \
    "\"$testbin\" -test.coverprofile=\"$covdir/\$OMPI_COMM_WORLD_RANK.out\" -test.run TestMPI"

awk '
  /^mode:/ { if (!mode) { print; mode=1 }; next }
  {
    key = $1; n = split($0, a, " ")
    if (key in cnt) { cnt[key] = (a[n] > cnt[key] ? a[n] : cnt[key]) }
    else            { cnt[key] = a[n]; stmt[key] = $1 " " a[2]; ord[++idx] = key }
  }
  END { for (i=1; i<=idx; i++) print stmt[ord[i]] " " cnt[ord[i]] }
' "$covdir"/*.out > "$covout"

go tool cover -func="$covout"
