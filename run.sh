#!/bin/sh

if [ -e /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
fi

PYTHONPATH="$(dirname $(readlink -f "$0")):$PYTHONPATH" python3 -m pycommflag "$@"

