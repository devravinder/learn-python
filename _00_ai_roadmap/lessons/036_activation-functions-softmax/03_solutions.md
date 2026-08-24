# 03 — Solutions: Activation Functions & Softmax

## 1–2. Activations and derivatives

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_prime(x): s = sigmoid(x); return s * (1 - s)

def tanh(x): return np.tanh(x)
def tanh_prime(x): return 1 - np.tanh(x)**2

def relu(x): return np.maximum(0, x)
def relu_prime(x): return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01): return np.where(x > 0, x, alpha * x)
def leaky_relu_prime(x, alpha=0.01): return np.where(x > 0, 1.0, alpha)

x = np.linspace(-10, 10, 200)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for f, name in [(sigmoid, "sigmoid"), (tanh, "tanh"), (relu, "relu"), (leaky_relu, "leaky_relu")]:
    axes[0].plot(x, f(x), label=name)
for f, name in [(sigmoid_prime, "sigmoid'"), (tanh_prime, "tanh'"), (relu_prime, "relu'"), (leaky_relu_prime, "leaky_relu'")]:
    axes[1].plot(x, f(x), label=name)
axes[0].legend(); axes[1].legend()
plt.show()
```

Sigmoid and tanh's derivative plots should visibly flatten to near 0 outside
roughly `[-3, 3]`; ReLU's derivative plot is a flat step (0 then 1) with no
decay at all for positive `x`.

## 3. Vanishing gradient simulation

```python
x = 3.0
sigmoid_chain = sigmoid_prime(x) ** 10
relu_chain = relu_prime(x) ** 10

print("sigmoid chained 10x:", sigmoid_chain)   # extremely small, e.g. ~1e-8
print("relu chained 10x:", relu_chain)          # exactly 1.0
```

The sigmoid-chained gradient shrinks to a vanishingly small number after
just 10 multiplications, while ReLU's stays exactly 1 — a direct numeric
demonstration of why deep sigmoid networks historically struggled to train
(gradients reaching early layers become negligibly small) while ReLU
networks don't have this specific problem.

## 4. Stable softmax

```python
def softmax(z):
    shifted = z - np.max(z)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum()

print(softmax(np.array([2.0, 1.0, 0.1])))
print(softmax(np.array([1000., 1001., 1002.])))   # no nan/inf
```

## 5. Temperature scaling

```python
z = np.array([2.0, 1.0, 0.5])
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for ax, T in zip(axes, [0.1, 0.5, 1.0, 2.0, 5.0]):
    probs = softmax(z / T)
    ax.bar(range(3), probs)
    ax.set_title(f"T={T}")
    ax.set_ylim(0, 1)
plt.show()
```

At `T=0.1`, the distribution should look nearly one-hot (almost all
probability on the highest logit); at `T=5.0`, it should look nearly
uniform across all three classes — confirming temperature's role in
sharpening vs flattening a probability distribution.

## 6. Sigmoid vs ReLU on a deeper network

```python
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
import time

X, y = make_moons(n_samples=300, noise=0.2, random_state=0)

for activation in ["logistic", "relu"]:
    t0 = time.time()
    model = MLPClassifier(hidden_layer_sizes=(20,20,20), activation=activation, max_iter=2000, random_state=0)
    model.fit(X, y)
    print(activation, "time:", time.time()-t0, "accuracy:", model.score(X, y), "iters:", model.n_iter_)
```

ReLU often converges in noticeably fewer iterations and/or reaches higher
final accuracy than sigmoid on a deeper (3-hidden-layer) network — consistent
with the vanishing-gradient explanation, though the exact gap depends on
initialization and this specific dataset's difficulty; the key qualitative
result (ReLU generally trains deep networks more easily than sigmoid) is the
important takeaway.
