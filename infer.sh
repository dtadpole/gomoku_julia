#!/bin/bash

while true ; do
    julia ./inf.jl "$@"
    sleep 1
done
