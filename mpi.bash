#!/bin/bash

set -e

platform='unknown'
unamestr=`uname`
if [[ "$unamestr" == 'Linux' ]]; then
   platform='linux'
elif [[ "$unamestr" == 'MINGW32_NT-6.2' ]]; then
   platform='windows'
elif [[ "$unamestr" == 'MINGW64_NT-10.0' ]]; then
   platform='windows'
elif [[ "$unamestr" == 'Darwin' ]]; then
   platform='darwin'
fi

echo "platform = $platform"

install_and_test(){
    HERE=`pwd`
    PKG=$1
    DOTEST=$2
    HASGENBASH=$3
    echo
    echo
    echo "=== compiling $PKG ============================================================="
    if [[ ! -z $HASGENBASH ]]; then
        bash xgenflagsfile.bash
    fi
    touch *.go
    go install
    if [ "$DOTEST" -eq 1 ]; then
        go test
    fi
    cd $HERE
}

if [[ $platform != 'windows' ]]; then
    install_and_test mpi 1 1
else
    install_and_test mpi 0
fi

echo
echo "=== SUCCESS! ============================================================"
