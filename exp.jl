include("./mcts.jl")

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

mutable struct Experiences

    # attributes
    _exp_list::Array{Tuple{Matrix{Float32},Matrix{Float32},Float32},1}
    _total_count::Int

    # functions
    length::Function
    totalCount::Function
    playGame::Function
    addExperience::Function
    sampleExperience::Function

    # constructor
    function Experiences()

        e = new(Array{Tuple{Matrix{Float32},Matrix{Float32},Float32},1}(), 0)

        """Length"""
        e.length = () -> length(e._exp_list)

        """Total size"""
        e.totalCount = () -> e._tot

        """Play game"""
        e.playGame = (model::Model) -> begin

            experiences = mcts_play_game(model)

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
            target_exp_size = max(args["experience_min"], min(args["experience_max"],
                floor(Int, (e._total_count - args["experience_min"]) / 2) + args["experience_min"]))

            # truncate experiences
            if length(e._exp_list) > target_exp_size
                e._exp_list = e._exp_list[length(e._exp_list)-target_exp_size+1:end]
            end
        end

        """Sample experience"""
        e.sampleExperience = (sample_size::Int) -> begin
            s, pi, v = unzip(sample(e._exp_list, sample_size))
            s = reshape(cat(s...; dims=3), (t.size(), t.size(), 1, sample_size))
            pi = reshape(cat(pi...; dims=3), (t.size() * t.size(), sample_size))
            v = reshape(cat(v...; dims=2), (1, sample_size))
            if args["model_cuda"] >= 0
                s = s |> gpu
                pi = pi |> gpu
                v = v |> gpu
            end
            return s, pi, v
        end

        return e
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    # create model
    model = Model(args["game_size"], channels=args["model_channels"])
    e = Experiences()
    e.playGame(model)
end
