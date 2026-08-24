# 01 — Concepts: Logistic Regression

## Why not just use linear regression for classification

Linear regression's output is unbounded (`-∞` to `∞`), but a class
probability needs to be in `[0, 1]`. Logistic regression fixes this by
squashing a linear combination through the **sigmoid** function (Lesson 013):

```
z = w·x + b
ŷ = sigmoid(z) = 1 / (1 + e^-z)
```

`ŷ` is interpreted as `P(class=1 | x)`. Classify as 1 if `ŷ >= 0.5`
(equivalently `z >= 0`), else 0.

## Decision boundary

Since the boundary is at `z = 0`, i.e. `w·x + b = 0`, logistic regression's
decision boundary is always a straight line (or hyperplane in higher
dimensions) — it's a **linear classifier**. It can't separate classes that
require a curved boundary without help (polynomial features, or a nonlinear
model like a tree or neural network).

## The loss function: binary cross-entropy

Using MSE here would work poorly (non-convex when combined with sigmoid,
weak gradient signal when confidently wrong). Instead, logistic regression
uses **binary cross-entropy** (Lesson 016, specialized to 2 classes):

```
L = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
```

For `y=1`: loss is `-log(ŷ)` (want `ŷ` close to 1). For `y=0`: loss is
`-log(1-ŷ)` (want `ŷ` close to 0). This loss is convex in the weights, so
gradient descent (Lesson 015) is guaranteed to find the global minimum.

## The gradient (turns out to be beautifully simple)

Despite the sigmoid and log, the gradient of binary cross-entropy with
respect to the weights simplifies to:

```
∂L/∂w = (ŷ - y) * x
```

— the same *form* as linear regression's gradient (Lesson 015)! This isn't
a coincidence: both are special cases of the **Generalized Linear Model**
framework, where matching a loss function to an appropriate link function
(sigmoid here) always produces this clean gradient form.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]   # P(class=1)
preds = model.predict(X_test)                # thresholded at 0.5 by default
```

## Odds and log-odds (logits)

`z = w·x + b` (the pre-sigmoid value) is called the **logit** or **log-odds**:

```
odds = P(y=1) / P(y=0) = P / (1-P)
logit = log(odds) = z
```

A one-unit increase in `x_i` changes the log-odds by exactly `w_i` — a
cleaner (if less intuitive) interpretation than "changes the probability by
`w_i`," since the sigmoid's nonlinearity means a fixed change in `z` shifts
probability by different amounts depending on where you start (a change
near `z=0` moves probability a lot; the same change near `z=5` moves it
almost none, since sigmoid saturates).

## Regularized logistic regression

Just like Lesson 022, L1/L2 penalties apply directly:
`LogisticRegression(penalty="l1"/"l2", C=...)` — note `C` is the *inverse*
of regularization strength here (smaller `C` = stronger regularization),
the opposite convention from Ridge/Lasso's `alpha`. Worth double-checking
whenever you switch between them.

## Multi-class: one-vs-rest or softmax

Logistic regression extends to multiple classes either by training one
binary classifier per class ("is it this class or not," picking the highest
score) or directly via **softmax regression** (multinomial logistic
regression) — the direct generalization covered fully in Lesson 036, where
sigmoid becomes softmax and binary cross-entropy becomes categorical
cross-entropy.
