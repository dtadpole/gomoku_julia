include("./model.jl")

mutable struct MctsNode
    _parent::Union{MctsNode,Nothing}
    _children::Array{MctsNode,1}
    _N::Int
    _P::Float32
    _Q::Float32
    _U::Float32
    _V::Float32
end

if abspath(PROGRAM_FILE) == @__FILE__
    m = Model(11)
    _play_model(m)
end
