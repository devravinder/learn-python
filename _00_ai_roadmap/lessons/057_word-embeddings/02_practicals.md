# 02 — Practicals: Word Embeddings

## From-scratch skip-gram (pure Python, no dependencies)

A tiny corpus deliberately built with two semantic clusters (pets vs
vehicles) that share sentence structure within each cluster:

```python
corpus = [
    "the cat sat on the mat",
    "the dog sat on the mat",
    "the cat ran in the park",
    "the dog ran in the park",
    "the car drove on the road",
    "the truck drove on the road",
    "the car parked on the street",
    "the truck parked on the street",
]
```

1. Build the vocabulary and generate skip-gram training pairs: for each
   word in each sentence, pair it with every other word within a window of
   2 positions on either side.

2. Implement a minimal skip-gram model **from scratch** (no `gensim`/
   `torch` needed — vocabulary is small enough for plain Python): an input
   embedding table `W_in` (one `D`-dim vector per word) and an output
   weight table `W_out`, trained via full softmax over the vocabulary and
   cross-entropy loss (Lesson 016) — predict the context word from the
   center word's embedding.

3. Train for 200 epochs with plain SGD (Lesson 015, no need for
   momentum/Adam at this tiny scale). Track average loss per epoch — does
   it decrease and plateau?

4. Compute cosine similarity (Lesson 010) between `W_in` vectors for:
   `cat`-`dog`, `car`-`truck` (same-cluster pairs) vs `cat`-`car`,
   `cat`-`truck`, `dog`-`car` (cross-cluster pairs). Do same-cluster pairs
   come out clearly more similar, purely from co-occurrence patterns, with
   no explicit "these are both animals" label anywhere in the training
   data?

5. Explain *why* `cat` and `dog` end up with near-identical embeddings
   here specifically — look at which other words appear in their
   immediate context across the corpus. Would `cat` and `dog` still end up
   similar if the corpus never put them in matching sentence structures?

## Pretrained embeddings (if you have `gensim` and can download ~100MB)

6. Load pretrained GloVe vectors (`gensim.downloader.load("glove-wiki-gigaword-100")`)
   and try `model.most_similar("king")`, then check the famous analogy
   `model.most_similar(positive=["king", "woman"], negative=["man"])` —
   does `"queen"` show up near the top? Also try an analogy that's known to
   work less cleanly (e.g. a profession-gender analogy) and reflect on why
   embeddings trained on real-world text can encode real-world biases
   present in that text.
