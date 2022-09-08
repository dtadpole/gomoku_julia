using Printf

mutable struct Game
    size::Int
    turn::Float32
    board::Array{Float32,2}
    last_move::Tuple{Int,Int}
end

function game_init(size::Int)
    g = Game(size, 1.0, zeros(Float32, size, size), (0, 0))
    return g
end

function game_display(g::Game)
    displayLine() = begin
        print("-")
        for j in 1:g.size
            print(" - -")
        end
        println()
    end
    for i in 1:g.size
        displayLine()
        print("|")
        for j in 1:g.size
            if g.board[i, j] > 0
                if (i, j) == g.last_move
                    print("(X)|")
                else
                    print(" X |")
                end
            elseif g.board[i, j] < 0
                if (i, j) == g.last_move
                    print("(O)|")
                else
                    print(" O |")
                end
            else
                print("   |")
            end
        end
        @printf(" %2d ", i)
        println()
    end
    displayLine()
    for i in 1:g.size
        @printf("  %c ", 'A' + i - 1)
    end
    println()
end

function game_play(g::Game, x::Int, y::Int)
    if g.board[x, y] == 0
        g.board[x, y] = g.turn
        g.turn = -g.turn
        g.last_move = (x, y)
    end
end

function game_clone(g::Game)
    g2 = Game(g.size, g.turn, copy(g.board), copy(g.last_move))
    return g2
end

function game_is_n_in_a_row(g::Game, row::Int, col::Int, dir_row::Int, dir_col::Int; n=5)
    """Check if there are n pieces in a row including (row, col) in the
    direction (dir_row, dir_col)."""
    count = 0
    color = row <= 0 || col <= 0 ? sign(g.turn) : sign(g.board[row, col])
    for i in -n+1:n-1
        curr_row = row + i * dir_row
        if curr_row <= 0 || curr_row > g.size
            continue
        end

        curr_col = col + i * dir_col
        if curr_col <= 0 || curr_col > g.size
            continue
        end

        # println("curr_row [$(curr_row)], curr_col [$(curr_col)], [$(g.board[curr_row])]")
        if g.board[curr_row, curr_col] == color
            count += 1
            if count >= n
                # println("row [$(row)], col [$(col)], dir_row [$(dir_row)], dir_col [$(dir_col)]")
                return true
            end
        else
            count = 0
        end
    end

    return false
end

function game_has_won(g::Game, row, col)
    """Check if the last move has won the game."""
    if game_is_n_in_a_row(g, row, col, 1, 0)
        return true
    elseif game_is_n_in_a_row(g, row, col, 0, 1)
        return true
    elseif game_is_n_in_a_row(g, row, col, 1, 1)
        return true
    elseif game_is_n_in_a_row(g, row, col, 1, -1)
        return true
    else
        return false
    end
end

function game_available_moves(g::Game)
    """Return a list of available moves."""
    moves = []
    for i in 1:g.size
        for j in 1:g.size
            if g.board[i, j] == 0
                push!(moves, (i, j))
            end
        end
    end
    return moves
end

function game_is_over(g::Game)
    """Check if the game is over."""
    if game_has_won(g, g.last_move...)
        return true
    end
    for i in 1:g.size
        for j in 1:g.size
            if g.board[i, j] == 0
                return false
            end
        end
    end
    return true
end

function game_score(g::Game)
    """Return the score of the game."""
    if game_has_won(g, g.last_move...)
        return -g.turn
    else
        return 0
    end
end

function game_playout(g::Game; display=false)
    """Play out a game and return the winner."""
    while !game_is_over(g)
        moves = game_available_moves(g)
        move = moves[rand(1:length(moves))]
        game_play(g, move...)
        if display
            game_display(g)
            sleep(0.2)
        end
    end
    game_display(g)
    score = game_score(g)
    println("score [$(score)]")
    return score
end

function game_test()
    g = game_init(11)
    # g.board = rand(-1.0:1.0, g.size, g.size)
    # game_display(g)
    game_playout(g, display=false)
end

if abspath(PROGRAM_FILE) == @__FILE__
    game_test()
end
