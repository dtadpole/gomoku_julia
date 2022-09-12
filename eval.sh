#!/bin/bash

while true ; do
    julia ./eval.jl "$@"
    sleep 15
done
