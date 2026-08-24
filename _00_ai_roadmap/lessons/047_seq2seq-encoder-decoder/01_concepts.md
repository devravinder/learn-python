# 01 — Concepts: Seq2Seq & Encoder-Decoder

## The problem: input and output sequences have different, variable lengths

Lesson 045's many-to-one/many-to-many (aligned) setups don't fit
translation: "I am happy" (3 words) might translate to a 2-word or 4-word
sentence in another language, with no fixed input-to-output position
alignment. **Seq2Seq** handles this with two separate RNNs (usually LSTMs,
Lesson 046): an **encoder** and a **decoder**.

## The architecture

```
Encoder: reads the entire input sequence, produces a final hidden
         state (and cell state, for LSTM) summarizing it — the "context vector"

Decoder: initialized with the encoder's final state, generates the
         output sequence one token at a time, feeding each generated
         token back in as the next input (autoregressive generation)
```

```python
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, c) = self.lstm(x)
        return h, c   # the "context" passed to the decoder

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h, c):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, (h, c))
        return self.fc(out), h, c
```

## Autoregressive generation at inference time

```python
h, c = encoder(input_seq)
decoder_input = torch.tensor([[SOS_TOKEN]])   # "start of sequence" token
outputs = []
for _ in range(max_len):
    logits, h, c = decoder(decoder_input, h, c)
    next_token = logits.argmax(dim=-1)
    if next_token.item() == EOS_TOKEN:        # "end of sequence"
        break
    outputs.append(next_token.item())
    decoder_input = next_token
```

**This exact loop — generate one token, feed it back in, repeat until an
end token or length limit** — is precisely how every LLM generates text
(Lesson 066), just with a Transformer decoder instead of an LSTM decoder.
If this loop makes sense here, GPT's generation loop will feel familiar
rather than new.

## Teacher forcing (training-time shortcut)

During training, instead of feeding the decoder's *own* (possibly wrong,
especially early in training) previous prediction back in, feed it the
**true** previous token from the target sequence instead — trains much
faster and more stably, since early mistakes don't compound and corrupt the
rest of the sequence during training. At inference time, there's no ground
truth to feed, so you must use the model's own generated tokens (as in the
loop above) — a real train/inference mismatch called **exposure bias**,
worth knowing exists even if not solved in this lesson.

## The bottleneck problem — the motivation for attention

The encoder must compress the **entire** input sequence into a single
fixed-size context vector (the final hidden state), regardless of whether
the input is 5 words or 500. For long sequences, this is an obvious
bottleneck: a single vector can't retain everything, so information from
early parts of a long input tends to get "forgotten" by the time the
decoder needs it — the same fundamental long-range problem from Lesson 045,
now specifically at the encoder-decoder handoff point rather than within a
single RNN.

**This exact bottleneck is what attention (Bahdanau et al. 2014, then
"Attention is All You Need" in 2017) was invented to fix**: instead of
forcing the encoder to compress everything into one vector, let the decoder
look back at **all** of the encoder's hidden states at every decoding step,
weighted by relevance — Lesson 058 picks up exactly here.

## BLEU score (brief mention — sequence generation evaluation)

Since Seq2Seq outputs (translations, summaries) rarely match a reference
exactly word-for-word, evaluation uses metrics like **BLEU** (n-gram
overlap with reference translations) rather than exact-match accuracy —
worth knowing exists; Lesson 073 covers generation evaluation more fully
for LLMs specifically.

## Why this lesson exists right before Module 8, not just Module 9-11

Seq2Seq's encoder-decoder pattern (and its exact bottleneck problem) is the
direct historical and conceptual lead-in to attention and Transformers.
Module 8 (Chess Bot / RL) doesn't need this directly, but placing it here
completes "the RNN family of ideas" as one coherent arc before switching to
reinforcement learning, then picking attention back up in Module 10 once
NLP foundations (Module 9) are in place.
