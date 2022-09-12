include("./game.jl")

using Flux
using CUDA
using JLD
using Serialization
using StatsBase

# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

function backup_file(filename)
    filename_bak = "$(filename).bak"
    filename_bak_bak = "$(filename_bak).bak"
    if isfile(filename_bak_bak)
        mv(filename_bak_bak, "$(filename_bak_bak).bak", force=true)
    end
    if isfile(filename_bak)
        mv(filename_bak, filename_bak_bak, force=true)
    end
    if isfile(filename)
        mv(filename, filename_bak, force=true)
    end
    return nothing
end


# model
mutable struct Model

    # attributes
    _size::Int
    _channels::Int
    _model::Chain

    # functions
    size::Function
    channels::Function
    forward::Function
    loss::Function
    save::Function
    load::Function
    clone::Function
    params::Function

    # constructor
    function Model(size::Int; channels=32)

        m = new(size, channels)

        """Size"""
        m.size = () -> m._size

        """Channels"""
        m.channels = () -> m._channels

        """Model"""
        m._model = Chain(
            Conv((3, 3), 1 => channels, relu; pad=(1, 1), bias=false),
            Conv((3, 3), channels => channels * 2, relu; pad=(1, 1), bias=false),
            Conv((3, 3), channels * 2 => channels * 4, relu; pad=(1, 1), bias=false),
            Conv((3, 3), channels * 4 => channels * 4, relu; pad=(1, 1), bias=false),
            Split(
                Chain(
                    Conv((1, 1), channels * 4 => div(channels, 2), relu),
                    x -> reshape(x, (div(channels, 2) * size * size, :)),
                    Dense(div(channels, 2) * size * size => size * size, relu),
                    softmax
                ),
                Chain(
                    Conv((1, 1), channels * 4 => div(channels, 2), relu),
                    x -> reshape(x, (div(channels, 2) * size * size, :)),
                    Dense(div(channels, 2) * size * size => channels * 2, relu),
                    Dense(channels * 2 => 1, tanh)
                )
            )
        )

        if args["model_cuda"] >= 0
            m._model = m._model |> gpu
        end


        """Forward pass"""
        m.forward = (x) -> begin
            p, v = m._model(x)
            return p, v
        end

        """Loss"""
        m.loss = (x, pi, v) -> begin
            p_, v_ = m.forward(x)
            # panelty() = args["model_loss_theta_entropy"] * sum(x -> sum(x -> x .* x, x), Flux.params(m._model))
            panelty() = -args["model_loss_coef_entropy"] * sum(p_ .* log.(p_), dims=1)
            loss = (v - v_) .^ 2 .- sum(pi .* log.(p_), dims=1) .+ panelty() # calculate loss for each sample in the batch
            return mean(loss) # reduce mean
        end

        """Save"""
        m.save = (path) -> begin
            io = open(path, "w")
            serialize(io, (m.size(), m.channels(), m._model))
            close(io)
        end

        """Load"""
        m.load = (path) -> begin
            io = open(path, "r")
            size, channels, model = deserialize(io)
            m._size = size
            m._channels = channels
            Flux.loadmodel!(m._model, model)
            close(io)
        end

        """Clone"""
        m.clone = () -> begin
            m2 = Model(m._size, channels=m._channels)
            m2._model = deepcopy(m._model)
            return m2
        end

        """Params"""
        m.params = () -> begin
            return Flux.params(m._model)
        end

        return m
    end
end

function _play_model(model::Model)
    # play game
    game = Game(model.size())
    while !game.is_over()
        if game.turn() > 0
            state = game.state()
        else
            state = -game.state()
        end
        if args["model_cuda"] >= 0
            state = state |> gpu
        end
        # get action
        pi, _ = model.forward(reshape(state, (model.size(), model.size(), 1, 1)))
        available_actions = game.available_actions()
        distribution = reshape(Array(pi), model.size()^2) .* available_actions
        action = sample(1:model.size()^2, Weights(distribution))
        # play action
        game.play_action(action)
        if args["game_display"]
            game.display()
            sleep(0.2)
        end
    end
    game.display()
    @info "Game score : $(game.score())"
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = Model(args["game_size"], channels=args["model_channels"])
    _play_model(m)
end
