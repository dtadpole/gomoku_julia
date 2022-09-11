#!/bin/bash

while true ; do
    julia ./infer.jl "$@"
    sleep 1
done
