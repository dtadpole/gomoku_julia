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
    rootDistribution::Function

    # constructor
    function MctsNode(game::Game, model::Model; ucb=3.0, gamma=0.98, parent=nothing, P=0.0)

        mcts = new(game, model, ucb, gamma, parent, nothing, 0, P, 0.0)

        """Size"""
        mcts.size = () -> mcts._game.size()

        """Parent"""
        mcts.parent = () -> mcts._parent

        """Clear parent"""
        mcts.clearParent = () -> begin
            mcts._parent = nothing
        end

        """Children"""
        mcts.children = () -> mcts._children

        """Expand children"""
        mcts.expandChildren = () -> begin
            if mcts._game.is_over()
                error("Cannot expand children - game is over")
            end
            pi, _ = mcts._model.forward(reshape(mcts._game.state(), (mcts.size(), mcts.size(), 1, 1)))
            Pa = reshape(pi, mcts.size()^2) .* mcts._game.available_actions()
            mcts._children = Array{Union{MctsNode,Nothing},1}(nothing, mcts.size()^2)
            for i in 1:mcts.size()^2
                if Pa[i] > 0.0
                    game = mcts._game.clone()
                    game.play_action(i)
                    mcts._children[i] = MctsNode(game, mcts._model, ucb=mcts._ucb, gamma=mcts._gamma, parent=mcts, P=Pa[i])
                end
            end
        end

        """Node Value : Q(S,a) + ucb * P(S,a) / (1 + N(S,a))"""
        mcts.nodeValue = () -> mcts._Q + mcts._ucb * mcts._P / (1 + mcts._N)

        """Update branch"""
        mcts.updateBranch = (v::Float32) -> begin
            mcts._Q += (mcts._Q * mcts._N + v) / (mcts._N + 1)
            mcts._N += 1
            if mcts._parent !== nothing
                mcts._parent.updateBranch(-v * mcts._gamma)
            end
        end

        """Travel branch"""
        mcts.travelBranch = () -> begin
            if mcts._game.is_over()
                mcts.updateBranch(mcts._game.score())
            else
                if mcts._children === nothing
                    mcts.expandChildren()
                end
                values = [mcts._children[i] === nothing ? -Inf : mcts._children[i].nodeValue() for i in 1:mcts.size()^2]
                i = argmax(values)
                mcts._children[i].travelBranch()
            end
        end

        """Root distribution"""
        mcts.rootDistribution = () -> begin
            pi = zeros(Float32, mcts.size()^2)
            for i in 1:mcts.size()^2
                if mcts._children[i] !== nothing
                    pi[i] = mcts._children[i]._N
                end
            end
            return pi ./ sum(pi)
        end

        return mcts

    end

end

function _mcts_play()

    # create game
    game = Game(7)

    # create model
    model = Model(7)

    # create node
    node = MctsNode(game, model, ucb=1.0, gamma=0.98)

    while !game.is_over()

        # play
        while node._N < 200
            node.travelBranch()
        end

        # get distribution
        pi = node.rootDistribution()

        # play action
        action = argmax(pi)
        # action = sample(1:model.size()^2, ProbabilityWeights(pi))
        game.play_action(action)

        # update node
        node = node._children[action]
        node.clearParent()

        # display
        game.display()

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    _mcts_play()
end
