# 03 — Solutions: LSTM & GRU

## 1. LSTM cell from scratch

```python
import numpy as np

rng = np.random.default_rng(0)
input_size, hidden_size = 3, 4
combined_size = input_size + hidden_size

def init_gate():
    return rng.normal(0, 0.3, (hidden_size, combined_size)), np.zeros(hidden_size)

Wf, bf = init_gate()
Wi, bi = init_gate()
Wo, bo = init_gate()
Wc, bc = init_gate()

def sigmoid(x): return 1 / (1 + np.exp(-x))

def lstm_cell(x_t, h_prev, c_prev):
    combined = np.concatenate([h_prev, x_t])
    f_t = sigmoid(Wf @ combined + bf)
    i_t = sigmoid(Wi @ combined + bi)
    o_t = sigmoid(Wo @ combined + bo)
    c_candidate = np.tanh(Wc @ combined + bc)
    c_t = f_t * c_prev + i_t * c_candidate
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t

sequence = rng.normal(size=(5, input_size))
h, c = np.zeros(hidden_size), np.zeros(hidden_size)
for t, x_t in enumerate(sequence):
    h, c = lstm_cell(x_t, h, c)
    print(t, "h:", h, "c:", c)
```

## 2–3. Long-range dependency: plain RNN vs LSTM vs GRU

```python
import torch
import torch.nn as nn

def make_long_range_data(n=500, seq_len=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, seq_len, 1)).astype(np.float32)
    y = (X[:, 0, 0] > 0).astype(np.float32).reshape(-1, 1)
    return torch.tensor(X), torch.tensor(y)

class SeqClassifier(nn.Module):
    def __init__(self, cell_type):
        super().__init__()
        cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = cls(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 1)
        self.cell_type = cell_type

    def forward(self, x):
        out = self.rnn(x)
        h_n = out[1][0] if self.cell_type == "LSTM" else out[1]
        return self.fc(h_n.squeeze(0))

def train_and_eval(cell_type, seq_len):
    X, y = make_long_range_data(seq_len=seq_len)
    torch.manual_seed(0)
    model = SeqClassifier(cell_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    preds = (torch.sigmoid(model(X)) > 0.5).float()
    return (preds == y).float().mean().item()

for seq_len in [50, 200]:
    print(f"--- sequence length {seq_len} ---")
    for cell_type in ["RNN", "LSTM", "GRU"]:
        acc = train_and_eval(cell_type, seq_len)
        print(cell_type, "accuracy:", acc)
```

At length 50, plain RNN may already lag behind LSTM/GRU on remembering the
first element; at length 200, the gap typically widens further — plain RNN
accuracy often degrades toward chance level (~50%) while LSTM/GRU retain
noticeably better accuracy, directly demonstrating the vanishing-gradient
gap growing with sequence length.

## 4. Parameter count comparison

```python
rnn = nn.RNN(10, 20)
gru = nn.GRU(10, 20)
lstm = nn.LSTM(10, 20)

rnn_params = sum(p.numel() for p in rnn.parameters())
gru_params = sum(p.numel() for p in gru.parameters())
lstm_params = sum(p.numel() for p in lstm.parameters())

print("RNN:", rnn_params)
print("GRU:", gru_params, "ratio to RNN:", gru_params / rnn_params)
print("LSTM:", lstm_params, "ratio to RNN:", lstm_params / rnn_params)
```

GRU should come out close to 3x the RNN's parameter count, LSTM close to
4x — directly matching the "3 gates vs 4 gate-equivalents vs 1 plain
transformation" structural difference from `01_concepts.md`.

## 5. Character-level LSTM text generator (preview of Lesson 063a)

```python
text = "the quick brown fox jumps over the lazy dog. " * 20
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}

data = torch.tensor([stoi[c] for c in text])
seq_len = 20

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        x = self.embed(x)
        out, state = self.lstm(x, state)
        return self.fc(out), state

model = CharLSTM(len(chars))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(200):
    i = torch.randint(0, len(data) - seq_len - 1, (32,))
    x_batch = torch.stack([data[j:j+seq_len] for j in i])
    y_batch = torch.stack([data[j+1:j+seq_len+1] for j in i])
    logits, _ = model(x_batch)
    loss = loss_fn(logits.reshape(-1, len(chars)), y_batch.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate
model.eval()
idx = torch.tensor([[stoi["t"]]])
state = None
generated = "t"
with torch.no_grad():
    for _ in range(100):
        logits, state = model(idx, state)
        probs = torch.softmax(logits[0, -1], dim=0)
        next_idx = torch.multinomial(probs, 1)
        generated += itos[next_idx.item()]
        idx = next_idx.unsqueeze(0)
print(generated)
```

On this tiny, highly repetitive training text, the model should learn to
reproduce word-like fragments and even full words from the training
sentence ("quick", "brown", "the") after enough training — a small,
concrete preview of exactly what Module 11's language modeling does at
vastly larger scale.

## 6. Depth vs training time and accuracy

```python
import time

for num_layers in [1, 3]:
    torch.manual_seed(0)
    class DeepLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 16, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(16, 1)
        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.fc(h_n[-1])

    X, y = make_long_range_data(seq_len=200)
    model = DeepLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for epoch in range(150):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - t0

    preds = (torch.sigmoid(model(X)) > 0.5).float()
    acc = (preds == y).float().mean().item()
    print(f"num_layers={num_layers}: time={elapsed:.2f}s, accuracy={acc:.3f}")
```

More layers increases training time roughly proportionally (more
sequential computation per timestep); on a task this simple (depends only
on remembering one early value), extra depth often doesn't meaningfully
improve accuracy over a single well-trained layer — depth helps more on
tasks requiring more complex, hierarchical sequence processing than "just
remember the first element."
