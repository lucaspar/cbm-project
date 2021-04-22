#!/bin/bash
for f in ./enumerated_graphs/*.g6
do
    ./showg_mac64 "$f" "${f/g6/adj}"
done
