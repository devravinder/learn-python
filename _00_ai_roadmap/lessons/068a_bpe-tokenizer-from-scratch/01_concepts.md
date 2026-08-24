# 01 — Concepts: A Complete BPE Tokenizer

## From a training-loop demo (Lesson 055/062) to a reusable tool

Earlier lessons trained BPE merges as a one-off demonstration. A real
tokenizer needs three separate, reusable operations:

```
train(text, vocab_size)  -> learns merges from a training corpus, once
encode(text)              -> applies learned merges to turn NEW text into token IDs
decode(ids)                -> turns token IDs back into text exactly
```

## The `Tokenizer` class

```python
class BPETokenizer:
    def __init__(self):
        self.merges = {}       # (id1, id2) -> new_id
        self.vocab = {i: bytes([i]) for i in range(256)}   # start: 256 raw bytes

    def train(self, text, vocab_size):
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

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._get_pair_counts(ids)
            # apply the EARLIEST-LEARNED applicable merge first (matches training order)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

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
```

## Why `encode` must apply merges in **the order they were learned**

This is the single most important subtlety beyond the training loop
itself: if merge `(A,B)->X` was learned *before* merge `(X,C)->Y`, encoding
new text must also apply `(A,B)` first — applying them out of order (or in
a different order than training) can produce a token sequence the model
never saw an equivalent of during training, silently degrading quality.
The `min(..., key=lambda p: self.merges.get(p, inf))` trick above finds
whichever present, mergeable pair was learned *earliest* (lowest merge
index/ID) at each step, guaranteeing training-consistent ordering.

## `decode` must handle arbitrary/partial byte sequences gracefully

Since BPE operates on raw bytes, a decoded ID sequence is guaranteed to
form *some* valid byte string, but if IDs are manipulated unusually
(unlikely in normal use, but worth defensive coding), the resulting bytes
might not form valid UTF-8 on their own. `errors="replace"` substitutes
the standard replacement character (`�`) rather than crashing — a
practical robustness detail real tokenizer implementations include.

## `vocab` construction: building meaning up from bytes

`self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]` builds
each merged token's actual byte representation by concatenating its two
parts' byte representations — recursively, since a merge can itself
combine two previously-merged tokens. This is what makes `decode` a simple
`vocab` lookup + concatenation, with no need to "unmerge" anything
step-by-step at decode time.

## Special tokens (added on top, not learned via merging)

Real tokenizers add fixed special tokens (`<|endoftext|>`, Lesson 062)
directly into the vocabulary at known, reserved IDs (outside the range
merges can produce), with dedicated handling in `encode` to detect them as
literal substrings before byte-level BPE ever runs on the surrounding text.
Not required for Project 013's scope, but worth knowing why real
tokenizers need this extra layer.

## What Project 013 does with this

Swap Lesson 064's character-level `encode`/`decode` for this class's
`encode`/`decode` (train it once on your training corpus, save
`self.merges`/`self.vocab` for reuse), keeping the rest of the GPT
architecture and training loop **completely unchanged** — a clean
demonstration that tokenization is a genuinely separable component from
the model itself, exactly as Lesson 062 argued.
