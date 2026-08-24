# 02 — Practicals: The Language Modeling Objective

## Pure Python: cross-entropy and perplexity by hand

1. Given a toy vocabulary of 4 tokens and a sequence of true next-token
   IDs `[1, 2, 0, 3]`, along with the model's predicted probability for
   the correct token at each position `[0.7, 0.4, 0.9, 0.2]` (assume the
   rest of the probability mass is spread elsewhere - you only need the
   probability of the *correct* token per position for cross-entropy),
   compute the average cross-entropy loss and the perplexity.

2. Compute perplexity for a "perfect" model (`P(correct)=1.0` for every
   position) and a "uniform random" model over a vocabulary of size 100
   (`P(correct)=0.01` for every position). Confirm the perfect model gives
   perplexity 1, and the uniform model's perplexity comes out to
   approximately the vocabulary size (100).

## The shift-by-one target mechanic

3. Given a token sequence `[5, 12, 7, 9, 3]` (a tokenized sentence, IDs
   only), construct the `(input, target)` pairs a language model would
   train on: `input = sequence[:-1]`, `target = sequence[1:]`. Print both
   and confirm `target[i]` is exactly `input[i+1]`'s value shifted --
   i.e., the model at position `i` is trained to predict what comes
   immediately after.

## PyTorch: cross-entropy loss on real logits

4. Create random logits of shape `(batch=2, seq_len=5, vocab_size=10)` and
   a random target of shape `(2, 5)` (`torch.randint(0, 10, (2,5))`).
   Compute the loss using `F.cross_entropy` after reshaping appropriately
   (per `01_concepts.md`). Compute perplexity from the loss.

5. Verify by hand: manually compute the cross-entropy loss for **one**
   specific (position, target) pair using `F.softmax` + indexing +
   `-log(...)`, and confirm it matches the corresponding per-position
   contribution `F.cross_entropy` would give (use `reduction="none"` to
   get per-position losses instead of the batch average, for an easy
   comparison).

6. Simulate training progress: generate 5 "checkpoints" of decreasing
   cross-entropy loss (e.g. `[3.5, 2.8, 2.1, 1.5, 1.0]`, roughly what a
   real training run's loss curve might look like early on). Convert each
   to perplexity and plot perplexity vs checkpoint. Does perplexity drop
   more steeply early on and flatten out later, even though loss itself
   might be decreasing at a more constant rate — why would an exponential
   relationship (`perplexity = exp(loss)`) produce that visual pattern?
