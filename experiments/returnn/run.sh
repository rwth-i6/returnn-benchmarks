#!/bin/bash

config=$1
test -e $config || {
	echo "usage: $0 <config>"
	exit 1
}

mydir=$(dirname $0)
$mydir/returnn/rnn.py $config

