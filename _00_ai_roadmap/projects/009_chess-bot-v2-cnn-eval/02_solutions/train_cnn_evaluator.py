"""Train a small CNN to imitate Project 008's classical evaluation
function (knowledge distillation), then report validation R².

Usage:
    python generate_training_data.py
    python train_cnn_evaluator.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).parent / "training_data.npz"
MODEL_PATH = Path(__file__).parent / "cnn_evaluator.pt"


class CNNEvaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # global average pooling (Lesson 044)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def main():
    data = np.load(DATA_PATH)
    X, y = data["X"], data["y"]

    # normalize labels (classical eval has a wide, mostly-material-driven range)
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    X_train, X_val, y_train, y_val = train_test_split(X, y_norm, test_size=0.15, random_state=0)
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(1)

    model = CNNEvaluator()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 128
    n = len(X_train_t)
    for epoch in range(30):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_train_t[idx]), y_train_t[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if epoch % 5 == 0:
            print(f"epoch {epoch}: val MSE (normalized) = {val_loss:.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).squeeze(1).numpy()
    r2 = r2_score(y_val, val_preds)
    print(f"\nValidation R² (predicting normalized classical eval): {r2:.3f}")

    torch.save({"model_state": model.state_dict(), "y_mean": y_mean, "y_std": y_std}, MODEL_PATH)
    print(f"Saved trained evaluator to {MODEL_PATH}")


if __name__ == "__main__":
    main()
