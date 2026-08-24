"""Chess Bot v1: negamax + alpha-beta search over a hand-crafted
evaluation function. Requires `pip install python-chess`.

Play interactively:
    python chess_bot.py

Run benchmarks/sanity checks instead:
    python chess_bot.py --benchmark
"""
import sys
import time

import chess

PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}

# Simplified piece-square tables (White's perspective; index 0 = a1, 63 = h8,
# matching python-chess's square numbering `rank*8 + file`). Mirrored via
# chess.square_mirror() for Black. Values are well-known simplified tables
# in the style popularized by Tomasz Michniewski's "Simplified Evaluation
# Function" - a standard, freely-reused starting point, not tuned here.
PAWN_TABLE = [
     0,  0,  0,   0,   0,  0,  0,  0,
     5, 10, 10, -20, -20, 10, 10,  5,
     5, -5,-10,   0,   0,-10, -5,  5,
     0,  0,  0,  20,  20,  0,  0,  0,
     5,  5, 10,  25,  25, 10,  5,  5,
    10, 10, 20,  30,  30, 20, 10, 10,
    50, 50, 50,  50,  50, 50, 50, 50,
     0,  0,  0,   0,   0,  0,  0,  0,
]

KNIGHT_TABLE = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PIECE_SQUARE_TABLES = {chess.PAWN: PAWN_TABLE, chess.KNIGHT: KNIGHT_TABLE}

MOBILITY_WEIGHT = 1.0
NODE_COUNT = 0


def evaluate(board):
    """Static evaluation, returned from the side-to-move's perspective
    (negamax convention, Lesson 048)."""
    if board.is_checkmate():
        return -99999   # side to move has been checkmated - as bad as it gets
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    white_perspective_score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        table = PIECE_SQUARE_TABLES.get(piece.piece_type)
        if table:
            idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            value += table[idx]
        white_perspective_score += value if piece.color == chess.WHITE else -value

    score = white_perspective_score if board.turn == chess.WHITE else -white_perspective_score

    # Simplification: mobility bonus only for the side to move (avoids the
    # extra cost of generating the opponent's moves via a null-move push).
    score += MOBILITY_WEIGHT * board.legal_moves.count()
    return score


def order_moves(board):
    """Captures first - a simple but effective move-ordering heuristic
    (Lesson 049) that substantially improves alpha-beta pruning."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    return moves


def negamax(board, depth, alpha, beta):
    global NODE_COUNT
    NODE_COUNT += 1
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    value = float("-inf")
    for move in order_moves(board):
        board.push(move)
        value = max(value, -negamax(board, depth - 1, -beta, -alpha))
        board.pop()
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return value


def find_best_move(board, depth):
    best_value = float("-inf")
    best_move = None
    for move in order_moves(board):
        board.push(move)
        value = -negamax(board, depth - 1, float("-inf"), float("inf"))
        board.pop()
        if value > best_value:
            best_value = value
            best_move = move
    return best_move


def play_interactive(depth=3):
    board = chess.Board()
    print(f"Chess Bot v1 (search depth={depth}). You are White.\n")
    while not board.is_game_over():
        print(board)
        print()
        if board.turn == chess.WHITE:
            move_str = input("Your move (SAN, e.g. e4, Nf3): ").strip()
            try:
                board.push_san(move_str)
            except ValueError:
                print("Illegal or unrecognized move, try again.\n")
                continue
        else:
            move = find_best_move(board, depth)
            print(f"Bot plays: {board.san(move)}\n")
            board.push(move)

    print(board)
    print("\nGame over:", board.outcome())


# --- benchmarks / sanity checks (01_requirement.md Q3, Q5, Q6) ---

def benchmark_move_ordering(depth=3):
    global NODE_COUNT
    board = chess.Board()

    NODE_COUNT = 0
    find_best_move(board, depth)
    ordered_nodes = NODE_COUNT

    # unordered: monkey-patch order_moves to not sort
    global order_moves
    original_order_moves = order_moves
    order_moves = lambda b: list(b.legal_moves)

    NODE_COUNT = 0
    find_best_move(board, depth)
    unordered_nodes = NODE_COUNT

    order_moves = original_order_moves
    print(f"depth={depth}  with capture-ordering: {ordered_nodes} nodes"
          f"  |  without ordering: {unordered_nodes} nodes")


def benchmark_depth_vs_depth(shallow=2, deep=4, games=10):
    results = {"deep_win": 0, "draw": 0, "shallow_win": 0}
    for g in range(games):
        board = chess.Board()
        deep_is_white = (g % 2 == 0)
        move_count = 0
        while not board.is_game_over() and move_count < 100:
            depth = deep if (board.turn == chess.WHITE) == deep_is_white else shallow
            move = find_best_move(board, depth)
            board.push(move)
            move_count += 1
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            results["draw"] += 1
        elif outcome.winner == deep_is_white:
            results["deep_win"] += 1
        else:
            results["shallow_win"] += 1
    print(f"depth={deep} vs depth={shallow} ({games} games): {results}")


def sanity_check_free_queen_capture():
    # White queen on d1 can capture a hanging black queen on d8 with a clear file.
    board = chess.Board("3q3k/8/8/8/8/8/8/3Q3K w - - 0 1")
    move = find_best_move(board, depth=2)
    print("Free queen capture position, bot plays:", board.san(move))
    assert board.san(move) in ("Qxd8+", "Qxd8"), "bot failed to take a free queen!"


def timing_by_depth():
    board = chess.Board()
    for depth in [1, 2, 3, 4]:
        t0 = time.time()
        find_best_move(board, depth)
        print(f"depth={depth}: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        print("=== Move ordering effect ===")
        benchmark_move_ordering(depth=3)
        print("\n=== Depth vs depth self-play ===")
        benchmark_depth_vs_depth(shallow=2, deep=4, games=10)
        print("\n=== Sanity check: free queen capture ===")
        sanity_check_free_queen_capture()
        print("\n=== Timing by depth ===")
        timing_by_depth()
    else:
        play_interactive(depth=3)
