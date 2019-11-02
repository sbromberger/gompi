#!/bin/bash

set -e

platform='invalid'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='valid'
elif [[ "$unamestr" == 'MINGW32_NT-6.2' ]]; then
   platform='invalid'
elif [[ "$unamestr" == 'MINGW64_NT-10.0' ]]; then
   platform='invalid'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='valid'
fi

echo "platform = $platform"

install_and_test(){
    HERE=`pwd`
    DOTEST=$1
    HASGENBASH=$2
    echo
    echo
    echo "=== compiling $PKG ==="
    if [[ ! -z $HASGENBASH ]]; then
        bash xgenflagsfile.bash
    fi
    touch *.go
    go generate
    go install
    if [ "$DOTEST" -eq 1 ]; then
        echo "=== testing $PKG ==="
        mpirun -n 4 go test
    fi
}

if [[ $platform != 'invalid' ]]; then
    install_and_test 1 1
else
    echo "=== This package is limited to Linux, UNIX, and OSX and cannot be installed on Windows."
    exit 1
fi

echo
echo "=== SUCCESS! ==="
