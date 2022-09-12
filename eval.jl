include("./mcts.jl")


function model_step(game::Game, model::Model)

    # create mcts node
    node = MctsNode(game, model; ucb=args["mcts_ucb"], gamma=args["mcts_gamma"])

    if game.is_over()
        @error "Game is over"
        return nothing
    end

    # mcts play
    while node._N < args["mcts_n"]
        node.travelBranch()
    end

    # policy distribution
    pa = reshape(node.rootN(), (model.size() * model.size()))
    if TEMPERATURE > 1e-4
        pa = softmax(log.(pa .+ 1e-8) / TEMPERATURE) .* game.available_actions()
        action = sample(1:model.size()^2, StatsBase.ProbabilityWeights(pa))
    else
        pa = pa .* game.available_actions()
        action = argmax(pa)
    end

    # play action
    game.play_action(action)
    if args["game_display"]
        @info maxk(node.rootN(), 5)
        # display
        game.display()
    end


    return nothing

end

function human_step(game::Game)

    if game.is_over()
        @error "Game is over"
        return nothing
    end

    # get action
    action = 0
    while action == 0
        print("What is your move : ")
        move = strip(uppercase(readline()))
        if move == ""
            continue
        end
        if move[1] in ('A':'Z')
            action = (parse(Int, move[2:end]) - 1) * game.size() + (move[1] - 'A') + 1
        else
            continue
        end
    end

    # play action
    game.play_action(action)
    if args["game_display"]
        # display
        game.display()
    end

    return nothing

end

function player_function(player::String)

    if player == "curr"

        filename = "./model/trained_$(args["game_size"])x$(args["game_size"]).curr"
        model = Model(args["game_size"], channels=args["model_channels"])
        model.load(filename)
        return (game) -> model_step(game, model), model

    elseif player == "best"

        filename = "./model/trained_$(args["game_size"])x$(args["game_size"]).best"
        model = Model(args["game_size"], channels=args["model_channels"])
        if isfile(filename)
            # load model if best file exist
            model.load(filename)
        end
        return (game) -> model_step(game, model), model

    elseif player == "human"

        return human_step, nothing

    else

        error("Unknown player")

    end
end

function eval_playout(player1_function::Function, player2_function::Function; turn=1.0)

    game = Game(args["game_size"], turn=turn)
    if args["game_display"]
        game.display()
    end

    if turn < 0
        player2_function(game)
    end

    while !game.is_over()

        if game.turn() > 0
            player1_function(game)
        else
            player2_function(game)
        end

    end

    if game.score() > 0
        @info "[ X ] wins !"
    elseif game.score() < 0
        @info "[ O ] wins !"
    else
        @info "Draw!"
    end

    return game.score()

end

function eval_run()

    player_1_wins = 0
    player_2_wins = 0
    draws = 0

    player1_function, model_1 = player_function(args["player_1"])
    player2_function, model_2 = player_function(args["player_2"])

    for i in 1:args["eval_playouts"]
        try
            @info "Evaluation Iteration [$(i)] - [$(args["player_1"])] vs [$(args["player_2"])]"
            score = eval_playout(player1_function, player2_function, turn=rand() > 0.5 ? 1.0 : -1.0)
            if score > 0
                player_1_wins += 1
            elseif score < 0
                player_2_wins += 1
            else
                draws += 1
            end
        catch e
            @error "Error" exception = (e, catch_backtrace())
            sleep(1.4^(i - 1))
        finally
            sleep(1)
        end
    end

    @info "[$(args["player_1"]) : $(player_1_wins)] vs [$(args["player_2"]) : $(player_2_wins)] - [$(draws) draws]"
    if args["player_1"] == "curr" && args["player_2"] == "best" && (player_1_wins - player_2_wins) >= args["eval_playouts"] / 2
        @info "Replacing best model with current model ..."
        backup_file("./model/trained_$(args["game_size"])x$(args["game_size"]).best")
        model_1.save("./model/trained_$(args["game_size"])x$(args["game_size"]).best")
        @info "Replaced best model with current model !"
    end

end


if abspath(PROGRAM_FILE) == @__FILE__
    eval_run()
end
