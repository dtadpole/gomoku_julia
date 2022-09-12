include("./mcts.jl")

using HTTP

rotate90 = (a::Matrix) -> begin
    reverse(transpose(a), dims=2)
end

rotate180 = (a::Matrix) -> begin
    reverse(reverse(a, dims=2), dims=1)
end

rotate270 = (a::Matrix) -> begin
    reverse(transpose(a), dims=1)
end

flip = (a::Matrix; dims = 2) -> begin
    reverse(a, dims=dims)
end

const _WEIGHTS = 1:floor(Int, args["exp_max"] * 2)

mutable struct Experiences

    # attributes
    _model::Model
    _exp_list::Array{Tuple{Matrix{Float32},Matrix{Float32},Float32},1}
    _total_count::Int

    # functions
    size::Function
    length::Function
    totalCount::Function
    playGame::Function
    addExperience::Function
    sampleExperience::Function
    startServer::Function

    # constructor
    function Experiences(model::Model)

        global _WEIGHTS

        e = new(model, Array{Tuple{Matrix{Float32},Matrix{Float32},Float32},1}(), 0)

        """Size"""
        e.size = () -> e._model.size()

        """Length"""
        e.length = () -> length(e._exp_list)

        """Total size"""
        e.totalCount = () -> e._total_count

        """Play game"""
        e.playGame = () -> begin

            experiences = mcts_play_game(e._model)

            for (state, pi, v) in experiences
                e.addExperience(state, pi, v)
            end

        end

        """Add experience"""
        e.addExperience = (s, pi, v) -> begin

            # flip and rotate per game symmetry
            s_flipped = flip(s)
            pi_flipped = flip(pi)
            push!(e._exp_list, (s, pi, v))
            push!(e._exp_list, (rotate90(s), rotate90(pi), v))
            push!(e._exp_list, (rotate180(s), rotate180(pi), v))
            push!(e._exp_list, (rotate270(s), rotate270(pi), v))
            push!(e._exp_list, (s_flipped, pi_flipped, v))
            push!(e._exp_list, (rotate90(s_flipped), rotate90(pi_flipped), v))
            push!(e._exp_list, (rotate180(s_flipped), rotate180(pi_flipped), v))
            push!(e._exp_list, (rotate270(s_flipped), rotate270(pi_flipped), v))

            # record total experience size
            e._total_count += 8

            # target experience size
            target_exp_size = max(args["exp_min"], min(args["exp_max"],
                floor(Int, (e._total_count - args["exp_min"]) / 2) + args["exp_min"]))

            # truncate experiences
            if length(e._exp_list) > target_exp_size
                e._exp_list = e._exp_list[length(e._exp_list)-target_exp_size+1:end]
            end
        end

        """Sample experience"""
        e.sampleExperience = (sample_size::Int) -> begin
            s, pi, v = unzip(sample(e._exp_list, ProbabilityWeights(_WEIGHTS[1:length(e._exp_list)]), sample_size))
            s = reshape(cat(s...; dims=3), (e.size(), e.size(), 1, sample_size))
            pi = reshape(cat(pi...; dims=3), (e.size() * e.size(), sample_size))
            v = reshape(cat(v...; dims=2), (1, sample_size))
            if args["model_cuda"] >= 0
                s = s |> gpu
                pi = pi |> gpu
                v = v |> gpu
            end
            return s, pi, v
        end

        """Start server"""
        e.startServer = () -> begin

            router = HTTP.Router()

            # post a single game experience
            HTTP.register!(router, "POST", "/experience", (req) -> begin
                body = HTTP.payload(req)
                io = IOBuffer(body)
                s, pi, v = deserialize(io)
                e.addExperience(s, pi, v)
                return HTTP.Response(200, "OK")
            end)

            # post a list of game experiences
            HTTP.register!(router, "POST", "/experiences", (req) -> begin
                body = HTTP.payload(req)
                io = IOBuffer(body)
                experiences = deserialize(io)
                for (s, pi, v) in experiences
                    e.addExperience(s, pi, v)
                end
                return HTTP.Response(200, "OK")
            end)

            # download model
            HTTP.register!(router, "GET", "/model", (req) -> begin
                io = IOBuffer()
                serialize(io, (e._model.size(), e._model.channels(), e._model._model))
                return HTTP.Response(200, Dict("Content-Type" => "application/octet-stream"), body=io.data)
            end)

            # start server async
            @async HTTP.serve(router, "0.0.0.0", args["exp_port"])
            @info "Experience server started on port [$(args["exp_port"])]"

        end

        return e

    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    # create model
    model = Model(args["game_size"], channels=args["model_channels"])
    e = Experiences(model)
    e.startServer()
    e.playGame()
end
