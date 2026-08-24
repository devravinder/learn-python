"""Generate (board_tensor, classical_eval_score) pairs by playing random
games and labeling positions with Project 008's hand-crafted evaluation
(knowledge distillation - see 01_requirement.md).

Usage:
    python generate_training_data.py
    -> writes training_data.npz (board tensors + labels) next to this script
"""
import random
from pathlib import Path

import chess
import numpy as np

# --- Project 008's evaluation function, reproduced here for a
# self-contained project (see ../../008_chess-bot-v1-minimax/02_solutions/chess_bot.py) ---
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}
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


def classical_evaluate(board):
    if board.is_checkmate():
        return -99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    white_score = 0
    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        table = PIECE_SQUARE_TABLES.get(piece.piece_type)
        if table:
            idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
            value += table[idx]
        white_score += value if piece.color == chess.WHITE else -value
    score = white_score if board.turn == chess.WHITE else -white_score
    return score + 1.0 * board.legal_moves.count()


PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board):
    """(12, 8, 8): one plane per (piece_type, color)."""
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        rank, file = chess.square_rank(square), chess.square_file(square)
        tensor[plane, rank, file] = 1.0
    return tensor


def generate_dataset(n_positions=10000, max_plies=40, seed=0):
    rng = random.Random(seed)
    tensors, labels = [], []
    while len(tensors) < n_positions:
        board = chess.Board()
        n_plies = rng.randint(0, max_plies)
        for _ in range(n_plies):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
        if board.is_game_over():
            continue   # skip terminal positions, evaluate() handles those separately in search
        tensors.append(board_to_tensor(board))
        labels.append(classical_evaluate(board))

    return np.array(tensors), np.array(labels, dtype=np.float32)


if __name__ == "__main__":
    X, y = generate_dataset(n_positions=10000)
    out_path = Path(__file__).parent / "training_data.npz"
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Wrote {len(X)} positions to {out_path}")
    print(f"label mean={y.mean():.1f}, std={y.std():.1f}, min={y.min():.1f}, max={y.max():.1f}")
