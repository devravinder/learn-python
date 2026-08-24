"""Chess Bot v2: identical search to Project 008, but evaluate() is a
trained CNN instead of the hand-crafted material+PST+mobility formula.

Usage:
    python generate_training_data.py
    python train_cnn_evaluator.py
    python chess_bot_v2.py               # play interactively
    python chess_bot_v2.py --benchmark    # v1 vs v2 head-to-head
"""
import sys
from pathlib import Path

import chess
import numpy as np
import torch

from generate_training_data import board_to_tensor, classical_evaluate
from train_cnn_evaluator import CNNEvaluator

MODEL_PATH = Path(__file__).parent / "cnn_evaluator.pt"

checkpoint = torch.load(MODEL_PATH, weights_only=False)
_model = CNNEvaluator()
_model.load_state_dict(checkpoint["model_state"])
_model.eval()
_Y_MEAN, _Y_STD = checkpoint["y_mean"], checkpoint["y_std"]


def cnn_evaluate(board):
    """Same signature/contract as Project 008's evaluate(): returns a score
    from the side-to-move's perspective."""
    if board.is_checkmate():
        return -99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0)
    with torch.no_grad():
        pred_norm = _model(tensor).item()
    return pred_norm * _Y_STD + _Y_MEAN   # undo the training-time normalization


def order_moves(board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    return moves


def negamax(board, depth, alpha, beta, eval_fn):
    if depth == 0 or board.is_game_over():
        return eval_fn(board)
    value = float("-inf")
    for move in order_moves(board):
        board.push(move)
        value = max(value, -negamax(board, depth - 1, -beta, -alpha, eval_fn))
        board.pop()
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return value


def find_best_move(board, depth, eval_fn):
    best_value, best_move = float("-inf"), None
    for move in order_moves(board):
        board.push(move)
        value = -negamax(board, depth - 1, float("-inf"), float("inf"), eval_fn)
        board.pop()
        if value > best_value:
            best_value, best_move = value, move
    return best_move


def play_interactive(depth=3):
    board = chess.Board()
    print(f"Chess Bot v2 (CNN evaluator, depth={depth}). You are White.\n")
    while not board.is_game_over():
        print(board, "\n")
        if board.turn == chess.WHITE:
            move_str = input("Your move (SAN): ").strip()
            try:
                board.push_san(move_str)
            except ValueError:
                print("Illegal move, try again.\n")
                continue
        else:
            move = find_best_move(board, depth, cnn_evaluate)
            print(f"Bot (CNN) plays: {board.san(move)}\n")
            board.push(move)
    print(board, "\n", board.outcome())


def validate_distillation():
    data = np.load(Path(__file__).parent / "training_data.npz")
    X, y = data["X"][-500:], data["y"][-500:]   # a held-out-ish slice for a quick spot check
    tensor = torch.tensor(X)
    with torch.no_grad():
        preds_norm = _model(tensor).squeeze(1).numpy()
    preds = preds_norm * _Y_STD + _Y_MEAN
    corr = np.corrcoef(preds, y)[0, 1]
    print(f"Correlation between CNN eval and classical eval on {len(y)} spot-check positions: {corr:.3f}")


def benchmark_v1_vs_v2(depth=3, games=10):
    results = {"v1_win": 0, "draw": 0, "v2_win": 0}
    for g in range(games):
        board = chess.Board()
        v2_is_white = (g % 2 == 0)
        moves = 0
        while not board.is_game_over() and moves < 100:
            eval_fn = cnn_evaluate if (board.turn == chess.WHITE) == v2_is_white else classical_evaluate
            move = find_best_move(board, depth, eval_fn)
            board.push(move)
            moves += 1
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            results["draw"] += 1
        elif outcome.winner == v2_is_white:
            results["v2_win"] += 1
        else:
            results["v1_win"] += 1
    print(f"v1 (classical) vs v2 (CNN), depth={depth}, {games} games: {results}")


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        validate_distillation()
        benchmark_v1_vs_v2()
    else:
        play_interactive()
