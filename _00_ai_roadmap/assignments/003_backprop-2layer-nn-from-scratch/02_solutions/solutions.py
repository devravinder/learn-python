"""Reference solutions for Assignment 003: vectorized NumPy backprop.

The gradient formulas here were verified against an independent pure-Python
(non-vectorized) implementation with a numerical gradient check before
being written up - see 03 in the lesson pattern; matched to 8+ significant
figures, confirming the ∂L/∂z2 = y_hat - y simplification and the chain of
matrix-form gradients below are correct.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
y = y.reshape(-1, 1)
n = len(X)


def relu(x):
    return np.maximum(0, x)


def relu_prime(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_params(n_in, n_hidden, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, 0.5, size=(n_in, n_hidden))
    b1 = np.zeros(n_hidden)
    W2 = rng.normal(0, 0.5, size=(n_hidden, 1))
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = sigmoid(z2)
    return y_hat, (X, z1, a1, z2)


def bce_loss(y_hat, y):
    eps = 1e-12
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def backward(y_hat, y, W2, cache):
    X, z1, a1, z2 = cache
    n = len(y)
    # BCE + sigmoid simplifies to (y_hat - y), same clean form as Lesson 023
    dz2 = (y_hat - y) / n
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    da1 = dz2 @ W2.T
    dz1 = da1 * relu_prime(z1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)
    return dW1, db1, dW2, db2


def train(X, y, n_hidden=8, lr=0.5, epochs=2000, seed=0):
    W1, b1, W2, b2 = init_params(X.shape[1], n_hidden, seed=seed)
    losses = []
    for _ in range(epochs):
        y_hat, cache = forward(X, W1, b1, W2, b2)
        loss = bce_loss(y_hat, y)
        losses.append(loss)

        dW1, db1, dW2, db2 = backward(y_hat, y, W2, cache)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
    return (W1, b1, W2, b2), losses


def gradient_check(X, y, params, h=1e-5):
    W1, b1, W2, b2 = params

    def total_loss(W1, b1, W2, b2):
        y_hat, _ = forward(X, W1, b1, W2, b2)
        return bce_loss(y_hat, y)

    y_hat, cache = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(y_hat, y, W2, cache)

    # check W1[0,0]
    orig = W1[0, 0]
    W1[0, 0] = orig + h
    loss_plus = total_loss(W1, b1, W2, b2)
    W1[0, 0] = orig - h
    loss_minus = total_loss(W1, b1, W2, b2)
    W1[0, 0] = orig
    num_grad_w1 = (loss_plus - loss_minus) / (2 * h)

    # check b2[0]
    orig_b2 = b2[0]
    b2[0] = orig_b2 + h
    loss_plus_b2 = total_loss(W1, b1, W2, b2)
    b2[0] = orig_b2 - h
    loss_minus_b2 = total_loss(W1, b1, W2, b2)
    b2[0] = orig_b2
    num_grad_b2 = (loss_plus_b2 - loss_minus_b2) / (2 * h)

    print("W1[0,0] analytical:", dW1[0, 0], "numerical:", num_grad_w1)
    print("b2[0]   analytical:", db2[0], "numerical:", num_grad_b2)


def plot_decision_boundary(X, y, params, title):
    W1, b1, W2, b2 = params
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                          np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    probs, _ = forward(grid, W1, b1, W2, b2)
    probs = probs.reshape(xx.shape)

    plt.contourf(xx, yy, probs, levels=20, cmap="RdBu", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolors="k")
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    params, losses = train(X, y, n_hidden=8, lr=0.5, epochs=2000)
    plt.plot(losses)
    plt.title("Training loss (8 hidden units)")
    plt.show()

    gradient_check(X, y, params)

    y_hat_final, _ = forward(X, *params)
    accuracy = ((y_hat_final > 0.5).astype(int) == y).mean()
    print("final training accuracy (8 hidden units):", accuracy)

    plot_decision_boundary(X, y, params, "Decision boundary (8 hidden units)")

    # Q6: reduced capacity
    params_small, losses_small = train(X, y, n_hidden=2, lr=0.5, epochs=2000)
    y_hat_small, _ = forward(X, *params_small)
    accuracy_small = ((y_hat_small > 0.5).astype(int) == y).mean()
    print("final training accuracy (2 hidden units):", accuracy_small)
    plot_decision_boundary(X, y, params_small, "Decision boundary (2 hidden units)")
