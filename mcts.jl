include("./model.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    m = Model(11)
    _play_model(m)
end
