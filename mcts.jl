include("./model.jl")

mutable struct MctsNode

    # attributes
    _game::Game
    _model::Model
    _cpuct::Float32
    _gamma::Float32
    _parent::Union{MctsNode,Nothing}
    _children::Union{Array{Union{MctsNode,Nothing},1},Nothing}
    _N::Int
    _P::Float32
    _Q::Float32
    # _U::Float32
    _Pi_V::Union{Tuple{Vector{Float32},Float32},Nothing}

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
    rootQ::Function

    # constructor
    function MctsNode(game::Game, model::Model; cpuct=3.0, gamma=0.98, parent=nothing, P=0.0)

        node = new(game, model, cpuct, gamma, parent, nothing, 0, P, 0.0, nothing)

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

        function _calc_Pi_V()
            if node._Pi_V === nothing
                state = node._game.state()
                if args["model_cuda"] >= 0
                    state = state |> gpu
                end
                pi, v = node._model.forward(reshape(state, (node.size(), node.size(), 1, 1)))
                node._Pi_V = (reshape(Array(pi), node.size()^2), Array(v)[1])
            end
            return node._Pi_V
        end

        """Expand children"""
        node.expandChildren = () -> begin
            if node._game.is_over()
                error("Cannot expand children - game is over")
            end
            # state = node._game.state()
            # if args["model_cuda"] >= 0
            #     state = state |> gpu
            # end
            # pi, _ = node._model.forward(reshape(state, (node.size(), node.size(), 1, 1)))
            pi, _ = _calc_Pi_V()
            Pa = pi .* node._game.available_actions()
            node._children = Array{Union{MctsNode,Nothing},1}(nothing, node.size()^2)
            for i in 1:node.size()^2
                if Pa[i] > 0.0
                    game = node._game.clone()
                    game.play_action(i)
                    node._children[i] = MctsNode(game, node._model; cpuct=node._cpuct, gamma=node._gamma, parent=node, P=Pa[i])
                end
            end
        end

        """Node Value : Q(S,a) + cpuct * P(S,a) * sqrt(sum(N(S,x) for x in siblings)) / (1 + N(S,a))"""
        node.nodeValue = () -> node._Q + node._cpuct * node._P * sqrt(node._parent._N) / (1 + node._N)

        """Update branch"""
        node.updateBranch = (v::Float32) -> begin
            node._Q = (node._Q * node._N + v) / (node._N + 1)
            node._N += 1
            if node._parent !== nothing
                node._parent.updateBranch(-v * node._gamma)
            end
        end

        """Travel branch"""
        node.travelBranch = (depth::Int = 1_000) -> begin
            if node._game.is_over()
                node.updateBranch(-node._game.score() * sign(node._game.turn()))
            elseif depth <= 0
                # state = node._game.state()
                # if args["model_cuda"] >= 0
                #     state = state |> gpu
                # end
                # _, v = node._model.forward(reshape(state, (node.size(), node.size(), 1, 1)))
                # node._V = Array(v)[1]
                _, v = _calc_Pi_V()
                node.updateBranch(v)
            else
                if node._children === nothing
                    node.expandChildren()
                end
                values = [node._children[i] === nothing ? -Inf : node._children[i].nodeValue() for i in 1:node.size()^2]
                i = argmax(values)
                node._children[i].travelBranch(depth - 1)
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

        """Root Q"""
        node.rootQ = () -> begin
            return node._Q
        end

        return node

    end

end

function maxk(a, k)
    b = partialsortperm(a, 1:k, rev=true)
    return collect(zip(b, a[b]))
end

function mcts_play_game(model_1::Model, model_2::Model)

    if model_1.size() != model_2.size()
        error("Models must have the same size")
    end

    SIZE = model_1.size()

    TEMPERATURE = args["mcts_temperature"]

    experiences = Array{Tuple{Array{Float32,2},Array{Float32,2},Float32},1}()

    # initialize game with random turn
    game_init_turn = rand() > 0.5 ? 1.0 : -1.0
    game = Game(SIZE, turn=game_init_turn)

    # create mcts node
    node_1 = MctsNode(game, model_1; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"])
    node_2 = MctsNode(game, model_2; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"])

    # choose which node to travel
    if game.turn() > 0
        node = node_1
        node_opposite = node_2
        model = model_1
        model_opposite = model_2
    else
        node = node_2
        node_opposite = node_1
        model = model_2
        model_opposite = model_1
    end

    while !game.is_over()

        if args["game_display"]
            start_time = time()
        end

        # play
        prior_N = node._N
        while node._N < args["mcts_n"]
            node.travelBranch(args["mcts_depth"])
        end
        post_N = node._N

        # get distribution
        pi = reshape(node.rootN(), (SIZE, SIZE))
        v = node.rootQ()
        push!(experiences, (game.state(), pi, v))

        # play action
        pa = reshape(pi, SIZE * SIZE)
        if TEMPERATURE > 1e-4
            pa = softmax(log.(pa .+ 1e-8) / TEMPERATURE) .* game.available_actions()
            # pa = softmax(1.0 / TEMPERATURE * log.(pa .+ 1e-8)) .* game.available_actions()
            action = sample(1:SIZE^2, ProbabilityWeights(pa))
        else
            pa = pa .* game.available_actions()
            action = argmax(pa)
        end

        game.play_action(action)
        if args["game_display"]
            duration = round(time() - start_time, digits=1)
            @info "$(join(["($(game.str_action(a)), $(n))" for (a, n) in maxk(node.rootN(), 5)], ", ")) [$(prior_N) -> $(post_N) : $(duration)s] "
            # display
            game.display()
        end

        # update mcts node
        node = node._children[action]
        node.clearParent()

        # update mcts node opposite
        if node_opposite._children !== nothing && node_opposite._children[action] !== nothing
            node_opposite = node_opposite._children[action]
            node_opposite.clearParent()
        else
            node_opposite = MctsNode(game, model_opposite; cpuct=args["mcts_cpuct"], gamma=args["mcts_gamma"])
        end

        # swap node and node_opposite
        node, node_opposite = node_opposite, node
        model, model_opposite = model_opposite, model

        # GC & reclaim CUDA memory
        GC.gc(true)
        if args["model_cuda"] >= 0
            CUDA.reclaim()
            # CUDA.memory_status()
            # println()
        end

    end

    return experiences, game_init_turn

end

if abspath(PROGRAM_FILE) == @__FILE__
    # create model
    model_1 = Model(args["game_size"], channels=args["model_channels"])
    model_2 = Model(args["game_size"], channels=args["model_channels"])
    mcts_play_game(model_1, model_2)
end
