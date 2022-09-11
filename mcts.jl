include("./model.jl")

mutable struct MctsNode

    # attributes
    _game::Game
    _model::Model
    _ucb::Float32
    _gamma::Float32
    _parent::Union{MctsNode,Nothing}
    _children::Union{Array{Union{MctsNode,Nothing},1},Nothing}
    _N::Int
    _P::Float32
    _Q::Float32
    # _U::Float32
    # _V::Float32

    # functions
    size::Function
    parent::Function
    clearParent::Function
    children::Function
    expandChildren::Function
    nodeValue::Function
    updateBranch::Function
    travelBranch::Function
    rootN::Function
    rootV::Function

    # constructor
    function MctsNode(game::Game, model::Model; ucb=5.0, gamma=0.98, parent=nothing, P=0.0)

        node = new(game, model, ucb, gamma, parent, nothing, 0, P, 0.0)

        """Size"""
        node.size = () -> node._game.size()

        """Parent"""
        node.parent = () -> node._parent

        """Clear parent"""
        node.clearParent = () -> begin
            node._parent = nothing
        end

        """Children"""
        node.children = () -> node._children

        """Expand children"""
        node.expandChildren = () -> begin
            if node._game.is_over()
                error("Cannot expand children - game is over")
            end
            state = node._game.state()
            if args["model_cuda"] >= 0
                state = state |> gpu
            end
            pi, _ = node._model.forward(reshape(state, (node.size(), node.size(), 1, 1)))
            Pa = reshape(Array(pi), node.size()^2) .* node._game.available_actions()
            node._children = Array{Union{MctsNode,Nothing},1}(nothing, node.size()^2)
            for i in 1:node.size()^2
                if Pa[i] > 0.0
                    game = node._game.clone()
                    game.play_action(i)
                    node._children[i] = MctsNode(game, node._model; ucb=node._ucb, gamma=node._gamma, parent=node, P=Pa[i])
                end
            end
        end

        """Node Value : Q(S,a) + ucb * P(S,a) / (1 + N(S,a))"""
        node.nodeValue = () -> node._Q + node._ucb * node._P / (1 + node._N)

        """Update branch"""
        node.updateBranch = (v::Float32) -> begin
            node._Q = (node._Q * node._N + v) / (node._N + 1)
            node._N += 1
            if node._parent !== nothing
                node._parent.updateBranch(-v * node._gamma)
            end
        end

        """Travel branch"""
        node.travelBranch = () -> begin
            if node._game.is_over()
                node.updateBranch(-node._game.score() * sign(node._game.turn()))
            else
                if node._children === nothing
                    node.expandChildren()
                end
                values = [node._children[i] === nothing ? -Inf : node._children[i].nodeValue() for i in 1:node.size()^2]
                i = argmax(values)
                node._children[i].travelBranch()
            end
        end

        """Root N distribution"""
        node.rootN = () -> begin
            pi = zeros(Int, node.size()^2)
            for i in 1:node.size()^2
                if node._children[i] !== nothing
                    pi[i] = node._children[i]._N
                end
            end
            return pi
        end

        """Root V"""
        node.rootV = () -> begin
            return node._Q
        end

        return node

    end

end

function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return collect(zip(b, a[b]))
end

function mcts_play_game(model::Model)

    TEMPERATURE = args["mcts_temperature"]

    experiences = Array{Tuple{Array{Float32,2},Array{Float32,2},Float32},1}()

    # initialize game
    game_turn = rand() > 0.5 ? 1.0 : -1.0
    game = Game(model.size(), turn=game_turn)

    # create mcts node
    node = MctsNode(game, model; ucb=args["mcts_ucb"], gamma=args["mcts_gamma"])

    while !game.is_over()

        # play
        while node._N < args["mcts_n"]
            node.travelBranch()
        end

        # get distribution
        pi = reshape(node.rootN(), (model.size(), model.size()))
        v = node.rootV()
        push!(experiences, (game.state(), pi, v))

        # play action
        pa = reshape(pi, model.size() * model.size())
        if TEMPERATURE > 1e-4
            pa = softmax(pa / TEMPERATURE) .* game.available_actions()
            # pa = softmax(1.0 / TEMPERATURE * log.(pa .+ 1e-8)) .* game.available_actions()
            action = sample(1:model.size()^2, StatsBase.ProbabilityWeights(pa))
        else
            pa = pa .* game.available_actions()
            action = argmax(pa)
        end

        game.play_action(action)
        if args["game_display"]
            # println("maxK : $(maxk(pa, 5))")
            @info maxk(node.rootN(), 5)
            # display
            game.display()
        end

        # update mcts node
        node = node._children[action]
        node.clearParent()

    end

    return experiences

end

if abspath(PROGRAM_FILE) == @__FILE__
    # create model
    model = Model(args["game_size"], channels=args["model_channels"])
    mcts_play_game(model)
end
