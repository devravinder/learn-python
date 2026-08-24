# 02 — Practicals: Seq2Seq & Encoder-Decoder

Build a toy "reverse the sequence" task — input a sequence of digits,
output them reversed (a stand-in for translation, small enough to train
quickly and easy to verify correctness by eye):

```python
import random

def make_example(min_len=3, max_len=7, vocab=range(10)):
    length = random.randint(min_len, max_len)
    seq = [random.choice(list(vocab)) for _ in range(length)]
    return seq, seq[::-1]
```

1. Build the vocabulary: digits 0-9 plus `SOS` (start), `EOS` (end), `PAD`
   tokens. Write an encoding function that converts a list of digits into a
   tensor of token ids, appending `EOS`.

2. Implement the `Encoder` and `Decoder` classes from `01_concepts.md`.
   Wire them into a training loop using **teacher forcing**: feed the
   decoder the true previous target token at each step (shifted target
   sequence, prefixed with `SOS`), compute cross-entropy loss against the
   true next token at every position.

3. Train on 2000 randomly generated reverse-sequence examples (pad to a
   fixed max length per batch, or use batch size 1 for simplicity) for
   enough epochs to see the loss drop substantially.

4. Implement autoregressive generation (per `01_concepts.md`'s inference
   loop, no teacher forcing) and test on 10 new random sequences — report
   what fraction are reversed *exactly* correctly.

5. Test on a sequence **longer** than anything seen in training (e.g.
   length 15 when training only used length 3-7). Does accuracy degrade —
   a direct, hands-on demonstration of the fixed-context-vector bottleneck
   from `01_concepts.md`?

6. Compare training **with** and **without** teacher forcing (i.e. feed the
   decoder its own previous prediction during training instead of the true
   token) on the same task. Does removing teacher forcing make training
   slower to converge or less stable, consistent with `01_concepts.md`'s
   explanation?
