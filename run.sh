#!/bin/sh
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
python3 -m pycommflag "$@"

