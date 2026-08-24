# 03 — Solutions: Word Embeddings

*(This code was actually run to produce the numbers below.)*

## 1. Vocabulary and training pairs

```python
corpus = [
    "the cat sat on the mat", "the dog sat on the mat",
    "the cat ran in the park", "the dog ran in the park",
    "the car drove on the road", "the truck drove on the road",
    "the car parked on the street", "the truck parked on the street",
]
sentences = [s.split() for s in corpus]
vocab = sorted(set(w for s in sentences for w in s))
word2idx = {w: i for i, w in enumerate(vocab)}
V = len(vocab)   # 15

WINDOW = 2
pairs = []
for s in sentences:
    for i, center in enumerate(s):
        for j in range(max(0, i-WINDOW), min(len(s), i+WINDOW+1)):
            if j != i:
                pairs.append((word2idx[center], word2idx[s[j]]))
print(len(pairs))   # 144
```

## 2–3. Minimal skip-gram, trained

```python
import math, random

D = 8
random.seed(0)
W_in = [[random.uniform(-0.5, 0.5) for _ in range(D)] for _ in range(V)]
W_out = [[random.uniform(-0.5, 0.5) for _ in range(D)] for _ in range(V)]

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

lr = 0.05
for epoch in range(200):
    random.shuffle(pairs)
    total_loss = 0.0
    for center, context in pairs:
        h = W_in[center]
        scores = [sum(h[d] * W_out[o][d] for d in range(D)) for o in range(V)]
        probs = softmax(scores)
        total_loss += -math.log(probs[context] + 1e-12)

        grad_scores = [probs[o] - (1.0 if o == context else 0.0) for o in range(V)]
        grad_h = [0.0] * D
        for o in range(V):
            g = grad_scores[o]
            for d in range(D):
                grad_h[d] += g * W_out[o][d]
                W_out[o][d] -= lr * g * h[d]
        for d in range(D):
            W_in[center][d] -= lr * grad_h[d]
    if epoch % 40 == 0:
        print(epoch, total_loss / len(pairs))
```

**Actual loss curve:** `2.676 -> 1.925 -> 1.928 -> 1.920 -> 1.921` (epochs
0, 40, 80, 120, 160) — drops sharply in the first 40 epochs, then plateaus,
consistent with this tiny, highly repetitive corpus having limited
information left to extract after the model learns the two clear clusters.

## 4. Same-cluster vs cross-cluster similarity

```python
def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(y*y for y in b))
    return dot / (na * nb)

def sim(w1, w2):
    return cos(W_in[word2idx[w1]], W_in[word2idx[w2]])

print("cat-dog:  ", sim("cat", "dog"))
print("car-truck:", sim("car", "truck"))
print("cat-car:  ", sim("cat", "car"))
print("cat-truck:", sim("cat", "truck"))
print("dog-car:  ", sim("dog", "car"))
```

**Actual output:**

```text
cat-dog:   0.999
car-truck: 0.998
cat-car:   0.175
cat-truck: 0.180
dog-car:   0.191
```

Same-cluster pairs (`cat`-`dog`, `car`-`truck`) land at **cosine similarity
≈ 0.999**, essentially identical vectors, while every cross-cluster pair
sits around **0.17-0.19** — a dramatic, clean separation, learned with
**zero explicit labels** saying "cat and dog are both animals." This is
the distributional hypothesis working exactly as `01_concepts.md`
describes, and a striking contrast with Lesson 056's TF-IDF result, where
"cat" and "dog" shared **no** signal at all (different tokens, no notion of
relatedness).

## 5. Why cat/dog end up nearly identical

In this corpus, `cat` and `dog` appear in **exactly the same sentence
templates** ("the ___ sat on the mat," "the ___ ran in the park") — every
context word `cat` sees, `dog` also sees, in identical positions. Since the
skip-gram objective pushes a word's embedding toward whatever lets it
predict its actual observed contexts, and `cat` and `dog` have *identical*
observed contexts here, their embeddings converge to nearly the same
point. If the corpus never placed them in matching structures (e.g. if
`dog` only ever appeared in unrelated sentences about, say, weather), there
would be no shared-context signal at all, and nothing would push their
embeddings together — the similarity is a direct, mechanical consequence of
shared context, not any deeper "understanding," which is worth
appreciating precisely because it explains both the power and the limits
of this approach.

## 6. Pretrained GloVe (run if you have the download)

```python
import gensim.downloader as api
model = api.load("glove-wiki-gigaword-100")

print(model.most_similar("king"))
print(model.most_similar(positive=["king", "woman"], negative=["man"]))
```

The analogy typically surfaces `"queen"` at or near the top of the
results — the well-known emergent vector-arithmetic property from
`01_concepts.md`. Profession-gender analogies (e.g.
`"doctor" - "man" + "woman"`) are documented in NLP fairness research to
sometimes surface stereotyped associations (e.g. skewing toward
"nurse") — a direct, real consequence of training on real-world text that
itself contains those statistical patterns/biases; the model isn't
reasoning about gender, it's reflecting co-occurrence statistics in its
training corpus exactly as literally as it reflects "king"/"queen."
