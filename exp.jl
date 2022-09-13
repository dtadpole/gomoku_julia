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
    _trained_count::Int

    # functions
    size::Function
    length::Function
    totalCount::Function
    trainedCount::Function
    addTrainedCount::Function
    playGame::Function
    addExperience::Function
    sampleExperience::Function
    startServer::Function
    save::Function
    load::Function

    # constructor
    function Experiences(model::Model)

        global _WEIGHTS

        e = new(model, Array{Tuple{Matrix{Float32},Matrix{Float32},Float32},1}(), 0, 0)

        """Size"""
        e.size = () -> e._model.size()

        """Length"""
        e.length = () -> length(e._exp_list)

        """Total count"""
        e.totalCount = () -> e._total_count

        """Trained count"""
        e.trainedCount = () -> e._trained_count

        """Add to trained count"""
        e.addTrainedCount = (count) -> begin
            e._trained_count += count
        end

        """Play game"""
        e.playGame = () -> begin

            experiences = mcts_play_game(e._model)

            for (state, pi, v) in experiences
                e.addExperience(state, pi, v)
            end

        end

        """Add experience"""
        e.addExperience = (s, pi, v) -> begin

            size_s = size(s)
            size_pi = size(pi)
            if size_s[1] != e.size() || size_s[2] != e.size() || size_pi[1] != e.size() || size_pi[2] != e.size()
                error("Invalid experience [s: $(size_s), pi: $(size_pi), v: $(v)]")
            end

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
                floor(Int, (e._total_count - args["exp_min"]) * args["exp_preserve_ratio"]) + args["exp_min"]))

            # truncate experiences
            if length(e._exp_list) > target_exp_size
                e._exp_list = e._exp_list[length(e._exp_list)-target_exp_size+1:end]
            end
        end

        """Sample experience"""
        e.sampleExperience = (sample_size::Int) -> begin
            s, pi, v = unzip(sample(e._exp_list, ProbabilityWeights(_WEIGHTS[1:min(length(e._exp_list), args["exp_max"])]), sample_size))
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
            @info "Run [infer.sh] to generate experiences ..."

        end

        """Save"""
        e.save = (path) -> begin
            io = open(path, "w")
            serialize(io, (e._total_count, e._trained_count, e._exp_list))
            close(io)
        end

        """Load"""
        e.load = (path) -> begin
            io = open(path, "r")
            total_count, trained_count, exp_list = deserialize(io)
            e._total_count = total_count
            e._trained_count = trained_count
            e._exp_list = exp_list
            close(io)
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
