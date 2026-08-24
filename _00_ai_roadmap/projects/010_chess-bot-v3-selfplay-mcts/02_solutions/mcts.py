"""Network-guided MCTS (PUCT selection, network value instead of random
rollouts) - Lesson 053 adapted per Lesson 054's AlphaZero-style approach.
"""
import math

import chess


class Node:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.prior = prior          # P(s,a) from the network, for the edge INTO this node
        self.children = {}          # move -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def puct_score(self, c_puct=1.5):
        parent_visits = self.parent.visit_count
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration


def expand(node, network):
    """Run the network once on this node's board, set children priors."""
    winner = _terminal_value(node.board)
    if winner is not None:
        node.expanded = True
        return winner   # value from the side-to-move's perspective at this (terminal) node

    priors, value = network.predict(node.board)
    for move, p in priors.items():
        child_board = node.board.copy()
        child_board.push(move)
        node.children[move] = Node(child_board, parent=node, prior=p)
    node.expanded = True
    return value


def _terminal_value(board):
    """Value from the side-to-move's perspective, or None if not terminal."""
    if board.is_checkmate():
        return -1.0   # side to move has been checkmated
    if board.is_game_over():
        return 0.0    # stalemate / draw by rule
    return None


def select_child(node, c_puct):
    return max(node.children.items(), key=lambda item: item[1].puct_score(c_puct))


def backpropagate(path, value):
    """`value` is from the leaf's side-to-move perspective; it flips sign
    at every level up the path, since each ancestor is the OPPONENT of the
    node below it (same alternation as negamax, Lesson 048)."""
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def run_mcts(root_board, network, n_simulations=100, c_puct=1.5):
    root = Node(root_board.copy())
    expand(root, network)

    for _ in range(n_simulations):
        node = root
        path = [node]
        while node.expanded and node.children:
            move, node = select_child(node, c_puct)
            path.append(node)

        value = expand(node, network)
        backpropagate(path, value)

    return root


def visit_distribution(root):
    """Normalized visit counts over root's children - the MCTS "improved
    policy" target used for training (Lesson 054)."""
    total = sum(child.visit_count for child in root.children.values())
    if total == 0:
        moves = list(root.children.keys())
        return {m: 1.0 / len(moves) for m in moves}
    return {move: child.visit_count / total for move, child in root.children.items()}


def select_move(root, temperature=1.0):
    """Sample a move from the visit distribution (temperature=1 during
    early self-play for exploration; temperature->0, i.e. argmax, for
    competitive play)."""
    dist = visit_distribution(root)
    moves = list(dist.keys())
    if temperature == 0:
        return max(dist, key=dist.get)
    import random
    weights = [dist[m] ** (1 / temperature) for m in moves]
    total = sum(weights)
    weights = [w / total for w in weights]
    return random.choices(moves, weights=weights, k=1)[0]
