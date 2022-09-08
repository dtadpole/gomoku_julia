include("./mcts.jl")


if abspath(PROGRAM_FILE) == @__FILE__
    _mcts_play()
end
