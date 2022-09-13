include("./mcts.jl")

using HTTP

global URL_BASE = ""

function download_model(id::Int)

    global URL_BASE

    url = "$(URL_BASE)/model/$(id)"

    r = HTTP.request(:GET, "$(url)")
    @info "[$(id)] Requesting model from [$(url)] : [status = $(r.status)]"
    io = IOBuffer(r.body)
    _id, _size, _channels, _model = deserialize(io)

    model = Model(_size, channels=_channels)
    Flux.loadmodel!(model._model, _model)
    # testmode!(model)
    params_size = sum([length(l) for l in Flux.params(model._model)])
    @info "[$(id)] Loaded model from [$(url)] : [$(model.size())x$(model.size()), channels=$(model.channels()), params=$(params_size)]"

    return _id, model

end

infer_playout = () -> begin

    global URL_BASE

    server = args["exp_server"]
    port = args["exp_port"]

    URL_BASE = "http://$server:$port"

    id_1, model_1 = download_model(1)
    id_2, model_2 = download_model(2)

    experiences, game_init_turn = mcts_play_game(model_1, model_2)

    id_first, id_second = game_init_turn > 0 ? (id_1, id_2) : (id_2, id_1)

    experiences_with_id = [(mod(step, 2) == 1 ? id_first : id_second, s, pi, v) for (step, (s, pi, v)) in enumerate(experiences)]

    io = IOBuffer()
    serialize(io, experiences_with_id)

    url = "$(URL_BASE)/experiences"
    r = HTTP.request(:POST, url, body=take!(io))
    @info "Posted experiences [$(url)] [len = $(length(experiences)), status = $(r.status)]"

end

if abspath(PROGRAM_FILE) == @__FILE__

    for i in 1:args["infer_playouts"]
        try
            @info repeat("-", 50)
            @info "Inference Iteration [$(i)] ..."
            @info repeat("-", 50)
            infer_playout()
        catch e
            @error "Error" exception = (e, catch_backtrace())
            sleep(1.1^(i - 1))
        finally
            sleep(1)
        end
    end

    @info repeat("-", 50)
    @info "Inference Completed [$(args["infer_playouts"])] Iterations ."
    @info repeat("-", 50)

end
