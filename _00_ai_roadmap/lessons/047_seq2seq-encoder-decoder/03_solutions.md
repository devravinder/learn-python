# 03 — Solutions: Seq2Seq & Encoder-Decoder

## 1. Vocabulary and encoding

```python
import random
import torch
import torch.nn as nn

PAD, SOS, EOS = 10, 11, 12
VOCAB_SIZE = 13

def encode(seq):
    return torch.tensor(list(seq) + [EOS])

def make_example(min_len=3, max_len=7):
    length = random.randint(min_len, max_len)
    seq = [random.randint(0, 9) for _ in range(length)]
    return seq, seq[::-1]
```

## 2. Encoder/Decoder with teacher forcing

```python
class Encoder(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        _, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, x, h, c):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, (h, c))
        return self.fc(out), h, c

encoder = Encoder()
decoder = Decoder()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

def train_step(src, tgt):
    src_t = encode(src).unsqueeze(0)
    tgt_t = encode(tgt)   # e.g. [t0, t1, ..., EOS]
    decoder_input = torch.cat([torch.tensor([SOS]), tgt_t[:-1]]).unsqueeze(0)

    h, c = encoder(src_t)
    logits, _, _ = decoder(decoder_input, h, c)
    loss = loss_fn(logits.squeeze(0), tgt_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 3. Training

```python
for step in range(3000):
    src, tgt = make_example()
    loss = train_step(src, tgt)
    if step % 500 == 0:
        print(step, loss)
```

Loss should drop substantially (often to near-zero cross-entropy) over a
few thousand steps on this simple, fully-deterministic task.

## 4. Autoregressive generation and accuracy

```python
def generate(src, max_len=10):
    src_t = encode(src).unsqueeze(0)
    h, c = encoder(src_t)
    decoder_input = torch.tensor([[SOS]])
    output = []
    for _ in range(max_len):
        logits, h, c = decoder(decoder_input, h, c)
        next_token = logits[0, -1].argmax().item()
        if next_token == EOS:
            break
        output.append(next_token)
        decoder_input = torch.tensor([[next_token]])
    return output

correct = 0
for _ in range(10):
    src, tgt = make_example()
    pred = generate(src)
    if pred == tgt:
        correct += 1
    print(src, "->", pred, "(expected", tgt, ")")
print(f"{correct}/10 exactly correct")
```

On this simple, fully-learnable task with enough training, most or all of
the 10 test sequences should reverse correctly.

## 5. Generalization to longer sequences

```python
long_src = [random.randint(0, 9) for _ in range(15)]
long_pred = generate(long_src, max_len=20)
print(long_src, "->", long_pred, "(expected", long_src[::-1], ")")
```

Accuracy typically degrades noticeably on sequences much longer than
anything seen in training — the fixed-size context vector (the encoder's
final hidden/cell state) has to compress more information than it was ever
trained to handle, exactly the bottleneck problem `01_concepts.md`
describes, and a concrete, hands-on reason attention (Lesson 058) was
invented.

## 6. Teacher forcing vs no teacher forcing

```python
def train_step_no_teacher_forcing(src, tgt):
    src_t = encode(src).unsqueeze(0)
    tgt_t = encode(tgt)
    h, c = encoder(src_t)

    decoder_input = torch.tensor([[SOS]])
    logits_list = []
    for t in range(len(tgt_t)):
        logits, h, c = decoder(decoder_input, h, c)
        logits_list.append(logits)
        decoder_input = logits.argmax(dim=-1).detach()   # feed back OWN prediction

    all_logits = torch.cat(logits_list, dim=1).squeeze(0)
    loss = loss_fn(all_logits, tgt_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# retrain fresh encoder/decoder with this function instead, compare loss curves
```

Without teacher forcing, training typically converges more slowly and less
stably — early in training, the decoder's own predictions are essentially
random, so it's being trained to predict correct continuations *given a
garbage prefix it generated itself*, a much harder and noisier learning
signal than always conditioning on the true prefix (teacher forcing). This
matches `01_concepts.md`'s explanation directly and is a real, measurable
effect, not just a theoretical concern.
