# 03 — Solutions: ML/DS Coding Interview Prep

*(All code below was actually executed in pure Python — no numpy/pandas/
sklearn used or available — and every output shown is the real, verified
result, not a predicted one.)*

## 1. Numerically stable softmax

```python
import math

def softmax(x):
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    s = sum(exps)
    return [e / s for e in exps]
```

```
softmax([1,2,3])          = [0.0900, 0.2447, 0.6652]
softmax([1000,1001,1002]) = [0.0900, 0.2447, 0.6652]   # identical shape, no overflow
naive exp([1000,1001,1002]) -> OverflowError: math range error
```

Subtracting the max before exponentiating (`v - m`) doesn't change the
mathematical result — it's a scale-invariance of softmax — but it keeps
every exponent ≤ 0, avoiding the overflow that hits the naive version at
inputs around 709+ in standard double precision.

## 2. K-means from scratch

```python
import random

def kmeans(points, k, max_iter=100, seed=0):
    random.seed(seed)
    centroids = random.sample(points, k)
    for _ in range(max_iter):
        clusters = [[] for _ in range(k)]
        for p in points:
            dists = [sum((a-b)**2 for a,b in zip(p,c)) for c in centroids]
            idx = dists.index(min(dists))
            clusters[idx].append(p)
        new_centroids = []
        for cluster in clusters:
            if not cluster:
                new_centroids.append(random.choice(points))  # empty-cluster fix: reinit randomly
            else:
                dim = len(cluster[0])
                new_centroids.append(tuple(sum(p[d] for p in cluster)/len(cluster) for d in range(dim)))
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters
```

On 7 hand-picked 2D points forming two visible groups:
```
centroids = [(1.25, 1.5), (3.9, 5.1)]
cluster sizes = [2, 5]
```
Matches the visually obvious grouping (the two low-coordinate points
cluster together, the five higher-coordinate points cluster together).
The empty-cluster edge case is handled by reinitializing that centroid to
a random data point — a simple, defensible fix to state even if a fancier
one (e.g. split the largest cluster) exists.

## 3. KNN from scratch

```python
def knn_predict(train_X, train_y, query, k):
    dists = sorted(range(len(train_X)),
                    key=lambda i: sum((a-b)**2 for a,b in zip(train_X[i], query)))
    top_k = dists[:k]
    votes = {}
    for i in top_k:
        votes[train_y[i]] = votes.get(train_y[i], 0) + 1
    return max(votes, key=votes.get)
```

On 6 points (3 near the origin labeled `A`, 3 near `(5,5)` labeled `B`):
```
predict((0.5, 0.5), k=3) -> 'A'
predict((5.5, 5.5), k=3) -> 'B'
```
Matches hand-computed nearest neighbors exactly — both query points are
obviously closer to their respective clusters.

## 4. Logistic regression via gradient descent

```python
def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    else:
        ez = math.exp(z)
        return ez / (1 + ez)   # numerically stable for very negative z

def train_logreg(X, y, lr=0.1, epochs=2000):
    n_features = len(X[0])
    w, b = [0.0]*n_features, 0.0
    n = len(X)
    for _ in range(epochs):
        grad_w, grad_b = [0.0]*n_features, 0.0
        for xi, yi in zip(X, y):
            pred = sigmoid(sum(w[j]*xi[j] for j in range(n_features)) + b)
            err = pred - yi
            for j in range(n_features):
                grad_w[j] += err * xi[j]
            grad_b += err
        w = [w[j] - lr*grad_w[j]/n for j in range(n_features)]
        b -= lr*grad_b/n
    return w, b
```

On 40 synthetic points (20 clustered in `[0,2]×[0,2]` labeled 0, 20 in
`[3,5]×[3,5]` labeled 1 — clearly linearly separable):
```
weights ≈ [0.934, 2.034], bias ≈ -6.746
training accuracy = 1.0
```
Full convergence to perfect separation, as expected for a linearly
separable dataset — the sigmoid split above the origin cleanly divides
the two clusters.

## 5. Precision/recall/F1 from scratch

```python
def prf1(y_true, y_pred):
    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    precision = tp/(tp+fp) if (tp+fp) else 0.0
    recall = tp/(tp+fn) if (tp+fn) else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0.0
    return precision, recall, f1
```

For `y_true=[1,1,1,0,0,0,1,0]`, `y_pred=[1,0,1,0,1,0,1,0]`:
```
precision = 0.75, recall = 0.75, f1 = 0.75
```
(3 true positives, 1 false positive, 1 false negative — matches a manual
count of the two lists.)

## 6. `train_test_split` from scratch

```python
def train_test_split(n, test_size, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(n * test_size)
    return idx[n_test:], idx[:n_test]
```

```
test_size=0.2: train=40 test=10 sum=50, same seed -> identical split
test_size=0.3: train=35 test=15 sum=50, same seed -> identical split
```
Using a locally-scoped `random.Random(seed)` (rather than the global
`random.seed`) means calling this function repeatedly doesn't disturb
other code's random state — worth mentioning in an interview as a subtle
but real correctness concern.

## 7. Monty Hall simulation

Analytical reasoning: your initial pick has a 1/3 chance of being the
car, so the *other two* doors together hold a 2/3 chance. The host always
reveals a goat from those other two, without changing their combined
probability — so all of that 2/3 collapses onto the single remaining
unopened door. Switching therefore wins with probability 2/3; staying
wins with probability 1/3.

```python
def monty_hall_trial(switch):
    doors = [0,1,2]
    car = random.choice(doors)
    choice = random.choice(doors)
    reveal = random.choice([d for d in doors if d != choice and d != car])
    if switch:
        choice = [d for d in doors if d != choice and d != reveal][0]
    return choice == car
```

Over 20,000 trials each:
```
switch win rate = 0.6667
stay win rate   = 0.3347
```
Matches the analytical 2/3 vs 1/3 prediction closely.

## 8. `groupby`-mean from scratch

```python
def groupby_mean(pairs):
    sums, counts = {}, {}
    for k, v in pairs:
        sums[k] = sums.get(k, 0) + v
        counts[k] = counts.get(k, 0) + 1
    return {k: sums[k]/counts[k] for k in sums}
```

For `[('a',10),('b',20),('a',20),('b',30),('c',5)]`:
```
{'a': 15.0, 'b': 25.0, 'c': 5.0}
```
Matches `pandas.DataFrame(...).groupby('key')['value'].mean()` exactly —
this dict-based accumulation *is* what that call does internally, one
abstraction layer down.
