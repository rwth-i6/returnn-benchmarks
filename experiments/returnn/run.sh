#!/bin/bash

config=$1

test -e "$config" || {
	echo "usage: $0 <config>"
	exit 1
}

#source theano-cuda-activate.sh

mydir=$(dirname $0)
set -x
$mydir/returnn/rnn.py $config ${@:2}
