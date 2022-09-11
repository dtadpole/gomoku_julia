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

        "--model_loss_coef_entropy"
        help = "model loss entropy coefficient"
        arg_type = Float32
        default = Float32(0.1)

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

        "--train_batch_num"
        help = "train batch number"
        arg_type = Int
        default = 16

        "--train_epochs"
        help = "train epochs"
        arg_type = Int
        default = 4

        "--adamw_beta1"
        help = "adamw beta1"
        arg_type = Float32
        default = Float32(0.9)

        "--adamw_beta2"
        help = "adamw beta2"
        arg_type = Float32
        default = Float32(0.999)

        "--adamw_weight_decay"
        help = "adamw weight decay"
        arg_type = Float32
        default = Float32(0.01)

        "--learning_rate"
        help = "learning rate"
        arg_type = Float32
        default = Float32(0.001)

        "--learning_rate_range"
        help = "learning rate range"
        arg_type = Float32
        default = Float32(5.0)

        "--learning_kl_target"
        help = "learning kl target"
        arg_type = Float32
        default = Float32(0.01)

        "--experience_min"
        help = "experience min"
        arg_type = Int
        default = 8_000

        "--experience_max"
        help = "experience max"
        arg_type = Int
        default = 400_000

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
