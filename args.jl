using ArgParse
using CUDA

function parse_commandline()

    s = ArgParseSettings()
    @add_arg_table s begin

        "--game_size"
        help = "game size"
        arg_type = Int
        default = 11

        "--game_display"
        help = "display game"
        arg_type = Bool
        default = true

        "--model_channels"
        help = "model channels"
        arg_type = Int
        default = 32

        "--model_cuda"
        help = "model cuda number"
        arg_type = Int
        default = -1

        "--mcts_n"
        help = "mcts play count"
        arg_type = Int
        default = 100

        "--mcts_ucb"
        help = "mcts ucb"
        arg_type = Float32
        default = Float32(5.0)

        "--mcts_gamma"
        help = "mcts gamma"
        arg_type = Float32
        default = Float32(0.98)

        "--mcts_temperature"
        help = "mcts temperature"
        arg_type = Float32
        default = Float32(1.5)

        "--train_batch_size"
        help = "train batch size"
        arg_type = Int
        default = 64

    end

    return parse_args(s)

end

args = parse_commandline()

if args["model_cuda"] >= 0
    CUDA.allowscalar(false)
    CUDA.device!(args["model_cuda"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    @info args
    @info args["game_size"]
    @info args["train_batch_size"]
end
