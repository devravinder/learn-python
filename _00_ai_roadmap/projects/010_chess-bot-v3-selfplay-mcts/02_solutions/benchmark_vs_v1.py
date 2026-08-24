"""Play the latest self-play generation (network-guided MCTS, no
hand-crafted evaluation) against Project 008's classical negamax bot.

Usage:
    python benchmark_vs_v1.py --generation 3 --games 10 --sims 200 --v1-depth 3
"""
import argparse
import sys
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "008_chess-bot-v1-minimax" / "02_solutions"))
from chess_bot import find_best_move as v1_find_best_move  # noqa: E402

from network import PolicyValueNet
from mcts import run_mcts, select_move

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def play_one_game(v3_net, v1_depth, v3_sims, v3_is_white, max_plies=150):
    board = chess.Board()
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        if (board.turn == chess.WHITE) == v3_is_white:
            root = run_mcts(board, v3_net, n_simulations=v3_sims)
            move = select_move(root, temperature=0.0)
        else:
            move = v1_find_best_move(board, v1_depth)
        board.push(move)
        ply += 1
    return board.outcome()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=int, default=3)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--v1-depth", type=int, default=3)
    args = parser.parse_args()

    net = PolicyValueNet()
    net.load_state_dict(torch.load(CHECKPOINT_DIR / f"gen{args.generation}.pt"))

    results = {"v3_win": 0, "draw": 0, "v1_win": 0}
    for g in range(args.games):
        v3_is_white = (g % 2 == 0)
        outcome = play_one_game(net, args.v1_depth, args.sims, v3_is_white)
        if outcome is None or outcome.winner is None:
            results["draw"] += 1
        elif outcome.winner == v3_is_white:
            results["v3_win"] += 1
        else:
            results["v1_win"] += 1

    print(f"Generation {args.generation} (self-play) vs v1 (classical, depth={args.v1_depth}): {results}")


if __name__ == "__main__":
    main()
