#!/bin/bash

while true ; do
    julia ./eval.jl --mcts_temperature 0.5 "$@"
    sleep 15
done
