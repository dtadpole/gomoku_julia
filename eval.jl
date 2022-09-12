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
    pa = pa .* game.available_actions()
    action = argmax(pa) # chooce max policy probability

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
        return (game) -> model_step(game, model)

    elseif player == "best"

        filename = "./model/trained_$(args["game_size"])x$(args["game_size"]).best"
        model = Model(args["game_size"], channels=args["model_channels"])
        if is_file(filename)
            # load model if best file exist
            model.load(filename)
        end
        return (game) -> model_step(game, model)

    elseif player == "human"

        return human_step

    else

        error("Unknown player")

    end
end

function play_game(player1::String, player2::String; turn=1.0)

    game = Game(args["game_size"], turn=turn)
    if args["game_display"]
        game.display()
    end

    player1_function = player_function(player1)
    player2_function = player_function(player2)

    while !game.is_over()

        if game.turn() * turn > 0
            player1_function(game)
        else
            player2_function(game)
        end

    end

end

if abspath(PROGRAM_FILE) == @__FILE__

    play_game(args["player_1"], args["player_2"], turn=rand() > 0.5 ? 1.0 : -1.0)

end
