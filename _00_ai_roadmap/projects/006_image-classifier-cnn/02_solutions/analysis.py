"""Reference solution: MLP vs small CNN on sklearn's digits dataset.

Run:
    python analysis.py
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

digits = load_digits()
X = digits.data / 16.0   # pixel values are 0-16 in this dataset; normalize to [0,1]
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


def make_loaders(X_train, y_train, X_test, y_test, shape_fn, batch_size=32):
    Xtr = shape_fn(torch.tensor(X_train, dtype=torch.float32))
    Xte = shape_fn(torch.tensor(X_test, dtype=torch.float32))
    ytr = torch.tensor(y_train, dtype=torch.long)
    yte = torch.tensor(y_test, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size)
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=30, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_true.extend(yb.tolist())
    accuracy = np.mean(np.array(all_preds) == np.array(all_true))
    return accuracy, np.array(all_true), np.array(all_preds)


# --- MLP baseline ---
mlp_train_loader, mlp_test_loader = make_loaders(
    X_train, y_train, X_test, y_test, shape_fn=lambda x: x
)
mlp = nn.Sequential(
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10),
)
mlp_acc, mlp_true, mlp_preds = train_model(mlp, mlp_train_loader, mlp_test_loader)
print("MLP test accuracy:", mlp_acc)
print("MLP confusion matrix:\n", confusion_matrix(mlp_true, mlp_preds))

# --- MLP + dropout ---
mlp_dropout = nn.Sequential(
    nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(32, 10),
)
mlp_dropout_acc, _, _ = train_model(mlp_dropout, mlp_train_loader, mlp_test_loader)
print("MLP + dropout test accuracy:", mlp_dropout_acc)

# --- small CNN ---
cnn_train_loader, cnn_test_loader = make_loaders(
    X_train, y_train, X_test, y_test, shape_fn=lambda x: x.reshape(-1, 1, 8, 8)
)
cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 10),
)
cnn_acc, cnn_true, cnn_preds = train_model(cnn, cnn_train_loader, cnn_test_loader)
print("CNN test accuracy:", cnn_acc)
print("CNN confusion matrix:\n", confusion_matrix(cnn_true, cnn_preds))

# --- error analysis on the best model ---
best_true, best_preds = (cnn_true, cnn_preds) if cnn_acc >= mlp_acc else (mlp_true, mlp_preds)
wrong_idx = np.where(best_true != best_preds)[0][:10]

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for ax, idx in zip(axes.ravel(), wrong_idx):
    img = X_test[idx].reshape(8, 8)
    ax.imshow(img, cmap="gray")
    ax.set_title(f"true={best_true[idx]} pred={best_preds[idx]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("misclassified.png")
plt.close()

print(f"\nSummary: MLP={mlp_acc:.3f}, MLP+dropout={mlp_dropout_acc:.3f}, CNN={cnn_acc:.3f}")
