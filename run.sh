#!/bin/sh

if [ -e /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
	export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
elif [ -e /usr/lib/libjemalloc.so.2 ]; then
	export LD_PRELOAD=/usr/lib/libjemalloc.so.2
fi

BASEDIR=$(dirname $(readlink -f "$0"))
if [ -e "$BASEDIR/venv/bin/activate" ]; then
	#echo "VENV $BASEDIR"
	. "$BASEDIR/venv/bin/activate"
fi

PYTHONPATH="$BASEDIR:$PYTHONPATH" exec python3 -m pycommflag "$@"

