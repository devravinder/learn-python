"""BPE tokenizer, reused directly from Lesson 068a."""
import json
from pathlib import Path


class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for i in range(num_merges):
            counts = self._get_pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (i + 1) % 100 == 0:
                print(f"  merge {i+1}/{num_merges}")
        return ids

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._get_pair_counts(ids)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        serializable = {f"{a},{b}": v for (a, b), v in self.merges.items()}
        Path(path).write_text(json.dumps(serializable))

    def load(self, path):
        serializable = json.loads(Path(path).read_text())
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for key, new_id in sorted(serializable.items(), key=lambda kv: kv[1]):
            a, b = map(int, key.split(","))
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]

    @staticmethod
    def _get_pair_counts(ids):
        counts = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, new_id):
        new_ids, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
