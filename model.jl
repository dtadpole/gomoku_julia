include("./game.jl")

using Flux
using CUDA
using BSON: @save, @load
using StatsBase

# custom split layer
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)


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

        if CUDA.functional()
            m._model = m._model |> gpu
        end


        """Forward pass"""
        m.forward = (x) -> begin
            if CUDA.functional()
                x = x |> gpu
            end
            p, v = m._model(x)
            return p, v
        end

        """Loss"""
        m.loss = (x, pi, v) -> begin
            p_, v_ = m.forward(x)
            panelty() = 0.001 * sum(x -> sum(x -> x .* x, x), Flux.params(m._model))
            loss = -sum(pi .* log.(p_)) + sum((v - v_) .^ 2) + panelty()
            return loss
        end

        """Save"""
        m.save = (path) -> begin
            _size = m.size()
            _channels = m.channels()
            _model = m._model
            @save path _size _channels _model
        end

        """Load"""
        m.load = (path) -> begin
            @load path _size _channels _model
            m._size = _size
            m._channels = _channels
            m._model = _model
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
        # get action
        pi, _ = model.forward(reshape(state, (model.size(), model.size(), 1, 1)))
        available_actions = game.available_actions()
        distribution = reshape(Array(pi), model.size()^2) .* available_actions
        action = sample(1:model.size()^2, ProbabilityWeights(distribution))
        # play action
        game.play_action(action)
        game.display()
        # print(length(model.params()))
        sleep(0.2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = Model(11)
    _play_model(m)
end
