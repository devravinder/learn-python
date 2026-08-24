"""Playable minimax/alpha-beta tic-tac-toe bot with depth-limited search
and a heuristic evaluation function (Assignment 004).

Play interactively:
    python tictactoe_bot.py

Run the analysis (Q3-Q4) instead:
    python tictactoe_bot.py --analyze
"""
import sys

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]

CALLS = 0


def legal_moves(board):
    return [i for i, c in enumerate(board) if c == " "]


def make_move(board, idx, player):
    b = board[:]
    b[idx] = player
    return b


def check_winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if " " not in board:
        return "draw"
    return None


def heuristic(board):
    """Q2: open winning lines for X minus open winning lines for O."""
    def open_lines(player):
        opponent = "O" if player == "X" else "X"
        return sum(1 for line in WIN_LINES if opponent not in [board[i] for i in line])
    return open_lines("X") - open_lines("O")


def alphabeta(board, player, alpha, beta, depth):
    global CALLS
    CALLS += 1
    winner = check_winner(board)
    if winner == "X":
        return 1000 + depth   # prefer faster wins, matching real engines' mate-distance preference
    if winner == "O":
        return -1000 - depth
    if winner == "draw":
        return 0
    if depth == 0:
        return heuristic(board)

    moves = legal_moves(board)
    if player == "X":
        value = float("-inf")
        for m in moves:
            value = max(value, alphabeta(make_move(board, m, "X"), "O", alpha, beta, depth - 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for m in moves:
            value = min(value, alphabeta(make_move(board, m, "O"), "X", alpha, beta, depth - 1))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def best_move(board, player, depth):
    scored = []
    for m in legal_moves(board):
        val = alphabeta(make_move(board, m, player), "O" if player == "X" else "X",
                         float("-inf"), float("inf"), depth - 1)
        scored.append((val, m))
    return max(scored)[1] if player == "X" else min(scored)[1]


def print_board(board):
    for r in range(3):
        row = board[r*3:r*3+3]
        print(" | ".join(c if c != " " else str(r*3+i) for i, c in enumerate(row)))
        if r < 2:
            print("-" * 9)


def play_interactive(bot_depth=9):
    board = [" "] * 9
    player = "X"
    print("You are O. Enter the number shown on the board to move.")
    print_board(board)
    while check_winner(board) is None:
        if player == "X":
            move = best_move(board, "X", bot_depth)
            print(f"\nBot plays {move}")
        else:
            move = int(input("\nYour move: "))
            while move not in legal_moves(board):
                move = int(input("Invalid move, try again: "))
        board = make_move(board, move, player)
        print_board(board)
        player = "O" if player == "X" else "X"

    winner = check_winner(board)
    print("\n" + ("Draw!" if winner == "draw" else f"{winner} wins!"))


def play_game(depth_x, depth_o):
    board = [" "] * 9
    player = "X"
    while check_winner(board) is None:
        depth = depth_x if player == "X" else depth_o
        move = best_move(board, player, depth)
        board = make_move(board, move, player)
        player = "O" if player == "X" else "X"
    return check_winner(board)


def analyze():
    global CALLS

    # Q3: depth-limited vs full-depth (9), 50 games each, alternating sides
    for test_depth in [1, 2]:
        results = {"depth_limited_win": 0, "draw": 0, "depth_limited_loss": 0}
        for game in range(50):
            dl_is_x = (game % 2 == 0)
            dx = test_depth if dl_is_x else 9
            do = 9 if dl_is_x else test_depth
            winner = play_game(dx, do)
            if winner == "draw":
                results["draw"] += 1
            elif (winner == "X") == dl_is_x:
                results["depth_limited_win"] += 1
            else:
                results["depth_limited_loss"] += 1
        print(f"depth={test_depth} vs full-depth (50 games): {results}")

    # Q4: call count vs depth
    board = [" "] * 9
    print("\ncalls to choose the first move, by depth limit:")
    for d in [1, 2, 3, 4, 5, 9]:
        CALLS = 0
        best_move(board, "X", d)
        print(f"  depth={d}: {CALLS} calls")


if __name__ == "__main__":
    if "--analyze" in sys.argv:
        analyze()
    else:
        play_interactive()
