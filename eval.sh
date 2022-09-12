#!/bin/bash

while true ; do
    julia ./eval.jl --mcts_temperature 0.33 "$@"
    sleep 15
done
