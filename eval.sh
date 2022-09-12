#!/bin/bash

while true ; do
    julia ./eval.jl "$@"
    sleep 1
done
