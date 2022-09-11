include("./mcts.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    # create model
    model = Model(args["game_size"], channels=args["model_channels"])
    mcts_play_game(model)
end
