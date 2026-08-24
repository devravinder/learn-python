# 03 — Solutions: Tokenization for LLMs

*(This code was actually run to produce the numbers below.)*

## 1. Byte encoding

```python
text = "the quick brown fox jumps over the lazy dog. the dog barks. the fox runs away quickly."
byte_seq = list(text.encode("utf-8"))
print(len(byte_seq), len(set(byte_seq)))   # 86 bytes, 28 unique values
```

28 unique byte values here, well under 256 — try it on text with emoji or
non-Latin scripts and it'll still never exceed 256, no matter what.

## 2–4. Byte-level BPE training

```python
from collections import Counter

def get_pair_counts(ids):
    counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts

def merge(ids, pair, new_id):
    new_ids, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

ids = byte_seq[:]
merges = []
next_id = 256
for step in range(15):
    counts = get_pair_counts(ids)
    best = max(counts, key=counts.get)
    ids = merge(ids, best, next_id)
    merges.append((best, next_id))
    print(f"merge {step+1}: {best} -> {next_id} (count={counts[best]})")
    next_id += 1

print("original:", len(byte_seq), "compressed:", len(ids), "ratio:", len(byte_seq)/len(ids))
```

**Actual output (first few merges, byte values shown as raw ints):**

```text
merge 1: (116, 104) -> 256 (count=4)   # 't','h' -> "th"
merge 2: (256, 101) -> 257 (count=4)   # "th",'e' -> "the"
merge 3: (257, 32)  -> 258 (count=4)   # "the",' ' -> "the "
merge 4: (32, 258)  -> 259 (count=3)   # ' ',"the " -> " the "
merge 5: (113, 117) -> 260 (count=2)   # 'q','u' -> "qu"
...
original: 86  compressed: 49  ratio: 1.755
```

The very first merges build up exactly `"the"` and then `" the "`
(bounded by spaces on both sides, appearing 3 times) — the single most
frequent short string in this text — precisely matching the
"most-frequent-pair-first" behavior demonstrated with characters in Lesson
055, now operating on raw bytes. After 15 merges, the text compresses from
86 bytes to 49 tokens, a **1.755x** compression ratio.

## 5. Generalization to new text

```python
def apply_merges(text, merges):
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges:
        ids = merge(ids, pair, new_id)
    return ids

new_text = "the dog runs quickly through the park"
new_ids = apply_merges(new_text, merges)
print(len(list(new_text.encode("utf-8"))), len(new_ids))
```

**Actual output: 37 bytes -> 22 tokens (1.68x compression)** — the merges
learned from the *training* paragraph (which include `" the "`, `"qu"`,
common short fragments) still apply usefully to this *different* sentence,
since it shares common English substrings, even though the exact sentence
was never seen during training. This is exactly how a real tokenizer,
trained once on a large corpus, generalizes to compress arbitrary new text
reasonably well.

## 6. Numeric tokenization artifacts

```python
import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("There are 1234567 people.")
for t in tokens:
    print(t, repr(enc.decode([t])))
```

Expect the number `1234567` to split into multiple chunks (e.g. something
like `"123"` + `"456"` + `"7"`, or a different split — the exact boundaries
depend on GPT-2's actual trained merge list, not a fixed rule) rather than
staying as one token or splitting into individual, clean digit groups.
This directly demonstrates why the *same* numeric value can tokenize
differently in different contexts (e.g. `"1234567"` vs `"11234567"` are not
guaranteed to share a common tokenization prefix) — a well-documented,
real cause of LLMs making surprising arithmetic mistakes, since the model
sees these as different, not-obviously-related token sequences rather than
directly perceiving "these are numbers that share digits."
