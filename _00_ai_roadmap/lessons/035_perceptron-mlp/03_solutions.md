# 03 — Solutions: Perceptron & Multi-Layer Perceptron

## 1. Perceptron on linearly separable data

```python
import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=0)

def train_perceptron(X, y, lr=0.1, epochs=50):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            z = np.dot(w, xi) + b
            pred = 1 if z > 0 else 0
            update = lr * (yi - pred)
            w += update * xi
            b += update
    return w, b

w, b = train_perceptron(X, y)
preds = ((X @ w + b) > 0).astype(int)
print("accuracy:", (preds == y).mean())   # should reach 1.0 on separable data
```

## 2. Perceptron failing on XOR

```python
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

w, b = train_perceptron(X_xor, y_xor, epochs=200)
preds = ((X_xor @ w + b) > 0).astype(int)
print("XOR accuracy:", (preds == y_xor).mean())   # stuck around 0.5-0.75, never 1.0
```

No setting of `w, b` can achieve 100% — confirmed by the fact that no
straight line separates the XOR classes (plot the 4 points to see it
directly: the two 1s and two 0s are diagonally opposite corners).

## 3. Hand-built MLP solving XOR

```python
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# hidden unit 1: fires (high) if at least one input is 1 -> roughly x1+x2
# hidden unit 2: fires (high) if both inputs are 1        -> roughly x1+x2-1.5, needs both
W1 = np.array([[1, 1], [1, 1]])       # shape (2 inputs, 2 hidden units)
b1 = np.array([-0.5, -1.5])
W2 = np.array([[1], [-2]])             # combine: h1 - 2*h2  (subtract off the "both" case)
b2 = np.array([-0.5])

def mlp_forward(x):
    h = relu(x @ W1 + b1)
    out = sigmoid(h @ W2 + b2)
    return out

for x in X_xor:
    print(x, mlp_forward(x))
```

With these hand-picked weights, `h1` activates for `(0,1)`, `(1,0)`, `(1,1)`
(sum >= 1), `h2` activates only for `(1,1)` (sum >= 2, minus 1.5 offset).
`h1 - 2*h2` is then positive only for the true XOR-true cases `(0,1)` and
`(1,0)`, and non-positive for `(0,0)` and `(1,1)` — solving XOR exactly with
2 hidden units and a nonlinearity, something no single-layer perceptron can
do.

## 4. Trainable MLP on XOR

```python
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(4,), activation="relu", max_iter=5000, random_state=0)
model.fit(X_xor, y_xor)
print(model.predict(X_xor), y_xor)
print("accuracy:", model.score(X_xor, y_xor))   # should reach 1.0
```

## 5. Minimum hidden units needed

```python
for size in [(1,), (2,), (4,), (10,)]:
    m = MLPClassifier(hidden_layer_sizes=size, max_iter=5000, random_state=0)
    m.fit(X_xor, y_xor)
    print(size, m.score(X_xor, y_xor))
```

A single hidden unit `(1,)` typically fails to reach 100% (one ReLU unit
still can't carve out XOR's non-convex "true" region alone); 2 hidden units
is usually the minimum that reliably succeeds — matching the hand-built
solution in Q3 exactly.

## 6. No nonlinearity fails regardless of depth

```python
model_linear = MLPClassifier(hidden_layer_sizes=(10, 10), activation="identity", max_iter=5000, random_state=0)
model_linear.fit(X_xor, y_xor)
print(model_linear.score(X_xor, y_xor))   # fails, same as a single perceptron
```

With `activation="identity"` (no nonlinearity), stacking any number of
layers is mathematically equivalent to a single linear layer — so this model
performs no better than the single perceptron from Q2, regardless of having
20 total hidden units across 2 layers. This is a direct, concrete
confirmation of why nonlinear activations (Lesson 036) are not optional.
