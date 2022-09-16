include("./args.jl")

using Printf

mutable struct Game

    # attributes
    _size::Int
    _turn::Float32
    _board::Array{Float32,2}
    _last_move::Tuple{Int,Int}
    _done::Bool

    # functions
    size::Function
    state::Function
    turn::Function
    available_moves::Function
    available_actions::Function
    play::Function
    play_action::Function
    str_action::Function
    is_over::Function
    score::Function
    has_won::Function
    display::Function
    clone::Function

    # constructor
    function Game(size::Int; turn=1.0)
        g = new(size, sign(turn), zeros(Float32, size, size), (0, 0), false)

        """Size"""
        g.size = () -> g._size

        """State"""
        g.state = () -> g._board

        """Turn"""
        g.turn = () -> g._turn

        """Available moves"""
        g.available_moves = () -> [(i, j) for i in 1:g._size for j in 1:g._size if g._board[i, j] == 0.0]

        """Available actions"""
        g.available_actions = () -> begin
            actions = zeros(Float32, g._size * g._size)
            for i in 1:g._size
                for j in 1:g._size
                    if g._board[i, j] == 0.0
                        actions[(i-1)*g._size+j] = 1.0
                    end
                end
            end
            return actions
        end

        """Play a move at (x, y)"""
        g.play = (x::Int, y::Int) -> begin
            if g._done
                error("Game is over")
            end
            if g._board[x, y] != 0.0
                error("Invalid move [$(x), $(y)]")
            end
            g._board[x, y] = sign(g._turn)
            g._last_move = (x, y)
            g._turn = -g._turn
            if g.has_won()
                g._done = true
            end
        end

        """Play an action"""
        g.play_action = (action::Int) -> begin
            x = div(action - 1, g._size) + 1
            y = mod(action - 1, g._size) + 1
            g.play(x, y)
        end

        """action string"""
        g.str_action = (action::Int) -> begin
            return "$(mod(action-1,size)+'A')$(div(action-1,size)+1)"
        end

        """Check if the game is over."""
        g.is_over = () -> begin
            if g._done
                return true
            end

            # check if the last move is a winning move
            if g.has_won()
                g._done = true
                return true
            end

            # check if there are any available moves
            if length(g.available_moves()) == 0
                g._done = true
                return true
            end

            return false
        end

        """Return the score of the game."""
        g.score = () -> begin
            if g.has_won()
                return -sign(g._turn)
            else
                return Float32(0.0)
            end
        end

        """Check if there are n pieces in a row including (row, col) in the
        direction (dir_row, dir_col)."""
        _is_n_in_a_row = (g::Game, row::Int, col::Int, dir_row::Int, dir_col::Int; n=5) -> begin
            count = 0
            color = row <= 0 || col <= 0 ? sign(g._turn) : sign(g._board[row, col])
            for i in -n+1:n-1
                curr_row = row + i * dir_row
                if curr_row <= 0 || curr_row > g._size
                    continue
                end

                curr_col = col + i * dir_col
                if curr_col <= 0 || curr_col > g._size
                    continue
                end

                if g._board[curr_row, curr_col] == color
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

        """Check if the last move has won the game."""
        g.has_won = () -> begin
            (row, col) = g._last_move
            if row <= 0 || col <= 0
                return false
            end
            if _is_n_in_a_row(g, row, col, 1, 0)
                return true
            elseif _is_n_in_a_row(g, row, col, 0, 1)
                return true
            elseif _is_n_in_a_row(g, row, col, 1, 1)
                return true
            elseif _is_n_in_a_row(g, row, col, 1, -1)
                return true
            else
                return false
            end
        end

        """Display the game board."""
        g.display = () -> begin
            displayLine() = begin
                print("    -")
                for j in 1:g._size
                    print(" - -")
                end
                println()
            end
            displayLegend() = begin
                print("    ")
                for i in 1:g._size
                    @printf("  %c ", 'A' + i - 1)
                end
                println()
            end
            displayLegend()
            for i in 1:g._size
                displayLine()
                @printf(" %-2d |", i)
                for j in 1:g._size
                    if g._board[i, j] > 0
                        if (i, j) == g._last_move
                            print("(X)|")
                        else
                            print(" X |")
                        end
                    elseif g._board[i, j] < 0
                        if (i, j) == g._last_move
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
            displayLegend()
        end

        """Clone the game."""
        g.clone = () -> begin
            g2 = Game(g._size, turn=g._turn)
            g2._board = copy(g._board)
            g2._last_move = g._last_move
            g2._done = g._done
            return g2
        end

        return g
    end
end


"""Play out a game and return the winner."""
function _random_playout(g::Game)
    while !g.is_over()
        moves = g.available_moves()
        move = moves[rand(1:length(moves))]
        g.play(move...)
        if args["game_display"]
            g.display()
            sleep(0.2)
        end
    end
    # game end
    g.display()
    score = g.score()
    println("score [$(score)]")
    return score
end

if abspath(PROGRAM_FILE) == @__FILE__
    g = Game(args["game_size"])
    _random_playout(g)
end
