include("./mcts.jl")

using Unzip

rotate90 = (a::Matrix) -> begin
    reverse(transpose(a), dims=2)
end

rotate180 = (a::Matrix) -> begin
    reverse(reverse(a, dims=2), dims=1)
end

rotate270 = (a::Matrix) -> begin
    reverse(transpose(a), dims=1)
end

flip = (a::Matrix; dims = 2) -> begin
    reverse(a, dims=dims)
end

mutable struct Train

    # attributes
    _model::Model
    _opt::Flux.Optimise.Optimiser

    _experiences::Array{Tuple{Array{Float32,2},Array{Float32,2},Float32},1}

    # functions
    size::Function
    playGame::Function
    addExperience::Function
    train::Function

    # constructor
    function Train(model::Model; opt=AdamW(0.0005, (0.9, 0.999), 1e-8))

        t = new(model, opt, Array{Tuple{Array{Float32,2},Array{Float32,2},Float32},1}())

        if args["model_cuda"] >= 0
            t._opt = t._opt |> gpu
        end

        """Size"""
        t.size = () -> t._model.size()

        """Play"""
        t.playGame = () -> begin

            experiences = mcts_play_game(t._model)

            for (state, pi, v) in experiences
                t.addExperience(state, pi, v)
            end

        end

        """Add experience"""
        t.addExperience = (s, pi, v) -> begin
            s_flipped = flip(s)
            pi_flipped = flip(pi)
            push!(t._experiences, (s, pi, v))
            push!(t._experiences, (rotate90(s), rotate90(pi), v))
            push!(t._experiences, (rotate180(s), rotate180(pi), v))
            push!(t._experiences, (rotate270(s), rotate270(pi), v))
            push!(t._experiences, (s_flipped, pi_flipped, v))
            push!(t._experiences, (rotate90(s_flipped), rotate90(pi_flipped), v))
            push!(t._experiences, (rotate180(s_flipped), rotate180(pi_flipped), v))
            push!(t._experiences, (rotate270(s_flipped), rotate270(pi_flipped), v))
        end

        """Train"""
        t.train = () -> begin

            BATCH_SIZE = args["train_batch_size"]

            exp_step = t.size() * t.size() * 8
            train_step = exp_step * 4

            exp_trained_count = 0
            exp_trained_weight = 0.0

            while true

                t.playGame()

                exp_len_step = floor(Int, length(t._experiences) / exp_step)

                exp_trained_step = floor(Int, exp_trained_weight / train_step)

                while exp_len_step > exp_trained_step

                    @info "Training started [$(exp_trained_count) / $(length(t._experiences))] ..."

                    # train
                    s, pi, v = unzip(sample(t._experiences, StatsBase.Weights(1:length(t._experiences)), BATCH_SIZE))
                    s = reshape(cat(s...; dims=3), (t.size(), t.size(), 1, BATCH_SIZE))
                    pi = reshape(cat(pi...; dims=3), (t.size() * t.size(), BATCH_SIZE))
                    v = reshape(cat(v...; dims=2), (1, BATCH_SIZE))
                    if args["model_cuda"] >= 0
                        s = s |> gpu
                        pi = pi |> gpu
                        v = v |> gpu
                    end
                    lossF = () -> t._model.loss(s, pi, v)
                    params = t._model.params()
                    # @info "typeof(params)" typeof(params)
                    grads = gradient(lossF, params)
                    Flux.Optimise.update!(t._opt, params, grads)

                    exp_trained_count += BATCH_SIZE
                    exp_trained_weight += BATCH_SIZE
                    exp_trained_step = floor(Int, exp_trained_weight / train_step)

                    @info "Training updated [$(exp_trained_count) / $(length(t._experiences))] ."

                end

                # save model
                t._model.save("model_$(model.size())_$(exp_trained_step).model")

            end
        end

        return t
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    model = Model(args["game_size"], channels=args["model_channels"])
    train = Train(model)
    train.train()
end
