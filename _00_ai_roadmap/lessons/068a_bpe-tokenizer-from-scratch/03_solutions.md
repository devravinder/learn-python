# 03 — Solutions: A Complete BPE Tokenizer

*(This code was actually run to produce every result below.)*

## 1–2. The tokenizer class, trained

```python
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
            if verbose:
                print(f"merge {i+1}: {pair} -> {new_id} ({self.vocab[new_id]})")
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

training_text = ("the quick brown fox jumps over the lazy dog. " * 10 +
                  "the dog barks at the fox. the fox runs quickly through the forest. " * 5)

tok = BPETokenizer()
final_ids = tok.train(training_text, vocab_size=280, verbose=True)
```

**Actual merges (selected, showing the buildup):**

```text
merge 1:  (116, 104) -> 256  (b'th')
merge 4:  (257, 258) -> 259  (b' the ')
merge 12: (266, 107) -> 267  (b'quick')
merge 17: (271, 103) -> 272  (b'dog')
merge 21: (275, 110) -> 276  (b'quick brown')
merge 23: (277, 270) -> 278  (b'quick brown fox ')
merge 24: (278, 106) -> 279  (b'quick brown fox j')
```

By merge 24, entire multi-word phrases (`"quick brown fox j"`) have been
absorbed into single tokens — later merges genuinely build on top of
earlier ones, exactly as `01_concepts.md` describes.

## 3. Round trip on training text

```python
encoded = tok.encode(training_text)
decoded = tok.decode(encoded)
print(decoded == training_text)      # True
print(encoded == final_ids)           # True
```

**Actual output: both `True`.** The decoded text matches exactly, and
`encode()` independently re-derives the identical token sequence `train()`
produced — confirming the merge-ordering logic in `encode()` is consistent
with training.

## 4. Round trip and compression on new text

```python
new_text = "the quick fox and the lazy dog run through the forest quickly."
encoded_new = tok.encode(new_text)
decoded_new = tok.decode(encoded_new)
print(decoded_new == new_text)   # True
print(len(new_text.encode("utf-8")) / len(encoded_new))   # compression ratio
```

**Actual output: round trip `True`; 62 bytes -> 35 tokens, a 1.77x
compression ratio** on text the tokenizer never saw verbatim during
training — the learned merges (common English fragments) still generalize.

## 5. Unicode stress test

```python
unicode_text = "café 🎉 日本語"
print(tok.decode(tok.encode(unicode_text)) == unicode_text)   # True
```

**Actual output: `True`.** Accented Latin characters, an emoji, and
Japanese characters all round-trip exactly — byte-level BPE's universal
coverage promise, verified directly rather than assumed.

## 6. More merges: training-text compression vs generalization

```python
for vs in [280, 320]:
    t = BPETokenizer()
    t.train(training_text, vocab_size=vs)
    ratio_train = len(training_text.encode("utf-8")) / len(t.encode(training_text))
    ratio_new = len(new_text.encode("utf-8")) / len(t.encode(new_text))
    print(vs, len(t.merges), ratio_train, ratio_new)
```

**Actual output:**

```text
vocab_size=280: 24 merges  train_ratio=2.39   new_text_ratio=1.77
vocab_size=320: 64 merges  train_ratio=16.35  new_text_ratio=1.77
```

A striking, clean result: nearly **3x more merges** (64 vs 24) pushes the
*training text's* compression from 2.39x to a dramatic 16.35x — but the
compression ratio on **genuinely new text stays exactly 1.77x, unchanged**.
The extra 40 merges are entirely absorbing this specific training text's
exact repeated sentences (an artifact of using highly repetitive text for
this demo) with **zero** benefit to new text — a real, measurable, and
fairly literal analogue to Lesson 017's overfitting: more "capacity"
(vocabulary/merges) fit to a small, repetitive training corpus, with no
corresponding gain in generalization. Real tokenizer training uses far
larger, more diverse corpora specifically to avoid this — a huge, varied
training set makes it much harder for merges to reflect anything other
than genuinely common, broadly-useful sub-word patterns.
