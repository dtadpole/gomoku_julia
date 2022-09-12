include("./mcts.jl")

using HTTP

infer_playout = () -> begin

    server = args["exp_server"]
    port = args["exp_port"]
    url = "http://$server:$port"

    r = HTTP.request(:GET, "$(url)/model")
    @info "Requesting model from $(url) : $(r.status)"
    io = IOBuffer(r.body)
    _size, _channels, _model = deserialize(io)

    model = Model(_size, channels=_channels)
    Flux.loadmodel!(model._model, _model)
    params_size = sum([length(l) for l in Flux.params(model._model)])
    @info "Requested model from $(url) : $(model.size()), $(model.channels()), $(params_size)"

    experiences = mcts_play_game(model)
    io = IOBuffer()
    serialize(io, experiences)

    r = HTTP.request(:POST, "$(url)/experiences", body=take!(io))
    @info "Posted experiences [$(length(experiences))] - [status : $(r.status)]"

end

if abspath(PROGRAM_FILE) == @__FILE__
    for i in 1:args["infer_playouts"]
        try
            @info "Iteration [$(i)]"
            infer_playout()
        catch e
            @error "Error" exception = (e, catch_backtrace())
            sleep(1.4^(i - 1))
        finally
            sleep(1)
        end
    end
end
