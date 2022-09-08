include("./game.jl")

using Flux
using BSON: @save, @load
using StatsBase

mutable struct Model
    # attributes
    _size::Int
    _channels::Int
    # conv layers
    _conv1::Conv
    _conv2::Conv
    _conv3::Conv
    _conv4::Conv
    # policy layers
    _pc::Conv
    _p1::Dense
    # value layers
    _vc::Conv
    _v1::Dense
    _v2::Dense
    # functions
    size::Function
    channels::Function
    forward::Function
    train::Function
    loss_P::Function
    loss_V::Function
    save::Function
    load::Function
    clone::Function
    # P::Function
    # V::Function

    # constructor
    function Model(size::Int; channels=32)

        m = new(size, channels)

        """Size"""
        m.size = () -> m._size

        """Channels"""
        m.channels = () -> m._channels

        """Convolution layers"""
        m._conv1 = Conv((3, 3), 1 => channels, relu; pad=(1, 1))
        m._conv2 = Conv((3, 3), channels => channels * 2, relu; pad=(1, 1))
        m._conv3 = Conv((3, 3), channels * 2 => channels * 4, relu; pad=(1, 1))
        m._conv4 = Conv((3, 3), channels * 4 => channels * 4, relu; pad=(1, 1))

        """Policy network"""
        m._pc = Conv((1, 1), channels * 4 => div(channels, 4), relu)
        m._p1 = Dense(div(channels, 4) * m._size * m._size => size * size, relu)
        # _p = Chain(_l1, _l2, _l3, _l4, _pc, _p1)
        # m.P = (x) -> p(x)

        """Value network"""
        m._vc = Conv((1, 1), channels * 4 => div(channels, 4), relu)
        m._v1 = Dense(div(channels, 4) * m._size * m._size => channels * 2, relu)
        m._v2 = Dense(channels * 2 => 1, relu)
        # _v = Chain(_l1, _l2, _l3, _l4, _vc, _v1, _v2)
        # m.V = (x) -> v(x)

        """Forward pass"""
        m.forward = (x) -> begin
            conv = m._conv4(m._conv3(m._conv2(m._conv1(x))))
            pc = reshape(m._pc(conv), (div(channels, 4) * m._size * m._size, :))
            # println("size of pc : ", size(pc))
            p = softmax(m._p1(pc))
            vc = reshape(m._vc(conv), (div(channels, 4) * m._size * m._size, :))
            # println("size of vc : ", size(vc))
            v = tanh.(m._v2(m._v1(vc)))
            return p, v
        end

        """Save"""
        m.save = (path) -> begin
            @save path _size _channels _conv1 _conv2 _conv3 _conv4 _pc _p1 _vc _v1 _v2
        end

        """Load"""
        m.load = (path) -> begin
            @load path _size _channels _conv1 _conv2 _conv3 _conv4 _pc _p1 _vc _v1 _v2
        end

        """Clone"""
        m.clone = () -> begin
            m2 = Model(m._size, channels=m._channels)
            # conv params
            m2._conv1.weight .= m._conv1.weight
            m2._conv1.bias .= m._conv1.bias
            m2._conv2.weight .= m._conv2.weight
            m2._conv2.bias .= m._conv2.bias
            m2._conv3.weight .= m._conv3.weight
            m2._conv3.bias .= m._conv3.bias
            m2._conv4.weight .= m._conv4.weight
            m2._conv4.bias .= m._conv4.bias
            # policy params
            m2._pc.weight .= m._pc.weight
            m2._pc.bias .= m._pc.bias
            m2._p1.weight .= m._p1.weight
            m2._p1.bias .= m._p1.bias
            # value params
            m2._vc.weight .= m._vc.weight
            m2._vc.bias .= m._vc.bias
            m2._v1.weight .= m._v1.weight
            m2._v1.bias .= m._v1.bias
            m2._v2.weight .= m._v2.weight
            m2._v2.bias .= m._v2.bias
            # return
            return m2
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
        p, _ = model.forward(reshape(state, (model.size(), model.size(), 1, 1)))
        available_actions = game.available_actions()
        distribution = reshape(p, (model.size() * model.size())) .* available_actions
        action = sample(1:model.size()^2, ProbabilityWeights(distribution))
        # play action
        game.play_action(action)
        game.display()
        sleep(0.2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = Model(11)
    # @show m
    _play_model(m)
end
