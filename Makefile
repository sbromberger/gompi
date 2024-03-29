.DEFAULT_GOAL := build

unamestr := $(shell uname)
INVALID=true
ifeq ($(unamestr), Linux)
INVALID=false
endif
ifeq ($(unamestr), Darwin)
INVALID=false
endif
ifeq ($(unamestr), FreeBSD)
INVALID=false
endif


CFLAGS=`mpicc -showme:compile`
LDFLAGS=`mpicc -showme:link`
FLAGS_FILE="xautogencgoflags.go"
	
define XGENFLAGS
// Copyright 2021 Seth Bromberger. All rights reserved.\n\
// Copyright 2016 The Gosl Authors. All rights reserved.\n\
// Use of this source code is governed by a BSD-style\n\
// license that can be found in the LICENSE file.\n\
// *** NOTE: this file was auto generated by Makefile ***\n\
// ***       and should be ignored                    ***\n\
\n\
// +build !windows\n\
\n\
package mpi\n\
/*\n\
#cgo CFLAGS: $(CFLAGS)  \n\
#cgo LDFLAGS: $(LDFLAGS)\n\
*/\n\
import \"C\"\n
endef

xgenflags:
	@echo "$(XGENFLAGS)" | sed "s/^ //g" >  $(FLAGS_FILE)

valid:
	@if [ "$(INVALID)" = "true" ]; then echo "Architecture $(unamestr) is not supported by this library"; exit 127; fi

gobuild:
	{ \
		touch *.go ; \
		go generate ; \
	}

goinstall: gobuild
	{ \
    	go install ; \
	}

build: valid xgenflags

install: build goinstall

test:
	mpirun -n 4 --oversubscribe go test . ;

clean:
	rm -f $(FLAGS_FILE)
