"""Self-play game generation, training, and generational iteration for
Chess Bot v3 (AlphaZero-style, Lesson 054).

Usage:
    python self_play.py --generations 3 --games 20 --sims 100
"""
import argparse
import random
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import PolicyValueNet, board_to_tensor, move_to_action, N_ACTIONS
from mcts import run_mcts, visit_distribution, select_move

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
MAX_GAME_PLIES = 120   # cap self-play games to keep them tractable at small scale


def self_play_game(network, n_sims, temperature_moves=10):
    board = chess.Board()
    examples = []   # (board_tensor, policy_target (dict move->prob), player_to_move)
    ply = 0
    while not board.is_game_over() and ply < MAX_GAME_PLIES:
        root = run_mcts(board, network, n_simulations=n_sims)
        dist = visit_distribution(root)
        temperature = 1.0 if ply < temperature_moves else 0.0
        move = select_move(root, temperature=temperature)

        examples.append((board_to_tensor(board), dist, board.turn))
        board.push(move)
        ply += 1

    if board.is_checkmate():
        winner = not board.turn   # the side that just moved delivered mate
    else:
        winner = None   # draw (incl. our ply cap - treated as a draw)

    training_data = []
    for tensor, dist, player in examples:
        policy_target = torch.zeros(N_ACTIONS)
        for move, prob in dist.items():
            policy_target[move_to_action(move)] = prob
        if winner is None:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0
        training_data.append((tensor, policy_target, z))
    return training_data


def train_on_examples(network, examples, epochs=5, batch_size=32, lr=1e-3):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
    tensors = torch.stack([e[0] for e in examples])
    policy_targets = torch.stack([e[1] for e in examples])
    value_targets = torch.tensor([e[2] for e in examples], dtype=torch.float32).unsqueeze(1)

    network.train()
    n = len(examples)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            policy_logits, value_pred = network(tensors[idx])
            policy_loss = -(policy_targets[idx] * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, value_targets[idx])
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        print(f"    epoch {epoch}: avg loss = {total_loss / n:.4f}")


def play_match(net_a, net_b, n_games=20, n_sims=100):
    """net_a vs net_b, alternating colors. Returns net_a's win rate."""
    wins_a, draws, wins_b = 0, 0, 0
    for g in range(n_games):
        board = chess.Board()
        a_is_white = (g % 2 == 0)
        ply = 0
        while not board.is_game_over() and ply < MAX_GAME_PLIES:
            net = net_a if (board.turn == chess.WHITE) == a_is_white else net_b
            root = run_mcts(board, net, n_simulations=n_sims)
            move = select_move(root, temperature=0.0)
            board.push(move)
            ply += 1
        if board.is_checkmate():
            winner_is_white = not board.turn
            if winner_is_white == a_is_white:
                wins_a += 1
            else:
                wins_b += 1
        else:
            draws += 1
    print(f"  net_a: {wins_a} wins, {draws} draws, {wins_b} losses (out of {n_games})")
    return wins_a / n_games


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    network = PolicyValueNet()
    gen0_state = {k: v.clone() for k, v in network.state_dict().items()}
    torch.save(gen0_state, CHECKPOINT_DIR / "gen0.pt")

    for gen in range(1, args.generations + 1):
        print(f"\n=== Generation {gen}: self-play ({args.games} games, {args.sims} sims/move) ===")
        all_examples = []
        for g in range(args.games):
            data = self_play_game(network, args.sims)
            all_examples.extend(data)
            print(f"  game {g+1}/{args.games}: {len(data)} positions")

        print(f"=== Generation {gen}: training on {len(all_examples)} positions ===")
        train_on_examples(network, all_examples)
        torch.save(network.state_dict(), CHECKPOINT_DIR / f"gen{gen}.pt")

        print(f"=== Generation {gen} vs Generation 0 (random init) ===")
        gen0_net = PolicyValueNet()
        gen0_net.load_state_dict(gen0_state)
        win_rate = play_match(network, gen0_net, n_games=min(args.games, 20), n_sims=args.sims)
        print(f"  Generation {gen} win rate vs Generation 0: {win_rate:.1%}")


if __name__ == "__main__":
    main()
