"""Reference solution: multiple linear regression via gradient descent,
entirely from scratch (NumPy only, no sklearn).

Run:
    python data/generate_data.py
    python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "housing.csv"
FEATURES = ["sqft", "bedrooms", "age", "distance_km"]


def load_and_standardize():
    df = pd.read_csv(DATA_PATH)
    X_raw = df[FEATURES].to_numpy()
    y = df["price"].to_numpy()

    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    X = (X_raw - means) / stds
    return X, y, means, stds


def mse(X, y, w, b):
    y_hat = X @ w + b
    return np.mean((y_hat - y) ** 2)


def gradients(X, y, w, b):
    n = len(y)
    y_hat = X @ w + b
    error = y_hat - y
    dw = (2 / n) * (X.T @ error)
    db = (2 / n) * error.sum()
    return dw, db


def train_batch(X, y, lr=0.1, epochs=2000, track_every=100):
    w = np.zeros(X.shape[1])
    b = 0.0
    losses = []
    for epoch in range(epochs):
        dw, db = gradients(X, y, w, b)
        w -= lr * dw
        b -= lr * db
        if epoch % track_every == 0:
            losses.append(mse(X, y, w, b))
    return w, b, losses


def train_minibatch(X, y, lr=0.1, epochs=200, batch_size=32, seed=0):
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    n = len(y)
    epochs_to_threshold = None
    threshold = 2.5e8  # roughly the converged MSE for this dataset
    for epoch in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            Xb, yb = X[batch_idx], y[batch_idx]
            dw, db = gradients(Xb, yb, w, b)
            w -= lr * dw
            b -= lr * db
        if epochs_to_threshold is None and mse(X, y, w, b) < threshold:
            epochs_to_threshold = epoch
    return w, b, epochs_to_threshold


def normal_equation(X, y):
    X_design = np.column_stack([np.ones(len(y)), X])
    theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    return theta[0], theta[1:]   # bias, weights


def main():
    X, y, means, stds = load_and_standardize()

    # --- 1 & 2: batch gradient descent + closed-form validation ---
    w, b, losses = train_batch(X, y, lr=0.1, epochs=2000)
    bias_closed, w_closed = normal_equation(X, y)

    print("=== Gradient descent vs normal equation ===")
    print("GD weights:    ", w.round(1), "bias:", round(b, 1))
    print("Closed-form:   ", w_closed.round(1), "bias:", round(bias_closed, 1))
    print("Match within 1%:", np.allclose(w, w_closed, rtol=0.01))

    plt.figure()
    plt.plot(losses)
    plt.title("Batch GD loss (lr=0.1)")
    plt.savefig(Path(__file__).parent / "loss_curve.png")
    plt.close()

    # --- 3: learning rate experiment ---
    plt.figure()
    for lr in [0.001, 0.01, 0.1]:
        _, _, ls = train_batch(X, y, lr=lr, epochs=1000)
        plt.plot(ls, label=f"lr={lr}")
    plt.legend()
    plt.yscale("log")
    plt.title("Learning rate comparison")
    plt.savefig(Path(__file__).parent / "lr_comparison.png")
    plt.close()

    # --- 4: batch vs mini-batch convergence speed ---
    _, _, epochs_mb = train_minibatch(X, y, lr=0.1, batch_size=32)
    print(f"\nMini-batch (size=32) reached target MSE in ~{epochs_mb} epochs")

    # --- 5: feature importance report ---
    print("\n=== Feature effect on price (standardized units) ===")
    for name, weight in sorted(zip(FEATURES, w), key=lambda t: -abs(t[1])):
        print(f"{name:12s} {weight:12.1f}")


if __name__ == "__main__":
    main()
