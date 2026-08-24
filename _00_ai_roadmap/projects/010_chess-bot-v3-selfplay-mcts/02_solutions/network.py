"""Combined policy+value network for Chess Bot v3 (AlphaZero-style).

Board encoding matches Project 009: (12, 8, 8), one plane per
(piece_type, color). Moves are encoded as from_square*64 + to_square
(4096 possible actions); underpromotion is not modeled - promotions
always default to a queen (a standard, documented simplification).
"""
import chess
import torch
import torch.nn as nn

N_ACTIONS = 64 * 64

PIECE_TO_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def board_to_tensor(board):
    tensor = torch.zeros(12, 8, 8)
    for square, piece in board.piece_map().items():
        plane = PIECE_TO_PLANE[(piece.piece_type, piece.color)]
        rank, file = chess.square_rank(square), chess.square_file(square)
        tensor[plane, rank, file] = 1.0
    return tensor


def move_to_action(move):
    return move.from_square * 64 + move.to_square


def action_to_move(board, action):
    """Decode an action index back into a legal chess.Move on `board`,
    adding a queen-promotion flag if the base (from,to) pair matches a
    pawn promotion (python-chess requires this to be explicit)."""
    from_sq, to_sq = divmod(action, 64)
    move = chess.Move(from_sq, to_sq)
    if move not in board.legal_moves:
        move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    return move


class PolicyValueNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, N_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Tanh(),
        )

    def forward(self, x):
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, board):
        """Returns (priors: dict[chess.Move -> float], value: float) for a
        single board, restricted + renormalized over legal moves only."""
        self.eval()
        x = board_to_tensor(board).unsqueeze(0)
        logits, value = self.forward(x)
        logits = logits.squeeze(0)

        legal = list(board.legal_moves)
        legal_actions = [move_to_action(m) for m in legal]
        mask = torch.full((N_ACTIONS,), float("-inf"))
        mask[legal_actions] = logits[legal_actions]
        probs = torch.softmax(mask, dim=0)

        priors = {m: probs[a].item() for m, a in zip(legal, legal_actions)}
        return priors, value.item()
