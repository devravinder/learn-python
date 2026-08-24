"""Reference solution: baselines + LSTM forecaster for daily sales.

Run:
    python data/generate_data.py
    python analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_PATH = Path(__file__).parent / "data" / "sales.csv"
TEST_DAYS = 60
WINDOW = 14


def mae(preds, actuals):
    return np.mean(np.abs(np.array(preds) - np.array(actuals)))


def rmse(preds, actuals):
    return np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2))


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    sales = df["sales"].to_numpy()
    n = len(sales)
    test_start = n - TEST_DAYS

    df["dow"] = df["date"].dt.day_name()
    print("=== Average sales by day of week ===")
    print(df.groupby("dow")["sales"].mean().sort_values(ascending=False))

    actuals = sales[test_start:]

    # --- baselines ---
    naive_preds = [sales[i - 1] for i in range(test_start, n)]
    seasonal_preds = [sales[i - 7] for i in range(test_start, n)]
    ma_preds = [sales[i - 7:i].mean() for i in range(test_start, n)]

    print("\n=== Baselines ===")
    for name, preds in [("naive", naive_preds), ("seasonal naive (lag-7)", seasonal_preds), ("moving avg(7)", ma_preds)]:
        print(f"{name:25s} MAE={mae(preds, actuals):.2f}  RMSE={rmse(preds, actuals):.2f}")

    # --- LSTM forecaster ---
    mean, std = sales[:test_start].mean(), sales[:test_start].std()   # fit scaler on TRAIN ONLY
    scaled = (sales - mean) / std

    def make_windows(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])
        return np.array(X), np.array(y)

    train_series = scaled[:test_start]
    X_train, y_train = make_windows(train_series, WINDOW)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

    class LSTMForecaster(nn.Module):
        def __init__(self, hidden_size=32):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h.squeeze(0))

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(300):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    # one-step-ahead forecast on the test period, using true history each step
    model.eval()
    lstm_preds = []
    with torch.no_grad():
        for i in range(test_start, n):
            window = scaled[i - WINDOW:i]
            x = torch.tensor(window, dtype=torch.float32).reshape(1, WINDOW, 1)
            pred_scaled = model(x).item()
            lstm_preds.append(pred_scaled * std + mean)   # back to original scale

    print(f"\n{'LSTM (1-step)':25s} MAE={mae(lstm_preds, actuals):.2f}  RMSE={rmse(lstm_preds, actuals):.2f}")

    # --- multi-step (7-day) autoregressive forecast ---
    horizon_errors = []
    history = list(scaled[:test_start])
    for start in range(test_start, n - 7, 7):
        window = history[start - WINDOW:start]
        step_preds = []
        with torch.no_grad():
            for step in range(7):
                x = torch.tensor(window[-WINDOW:], dtype=torch.float32).reshape(1, WINDOW, 1)
                pred_scaled = model(x).item()
                step_preds.append(pred_scaled)
                window.append(pred_scaled)
        step_preds_original = [p * std + mean for p in step_preds]
        step_actuals = sales[start:start+7]
        horizon_errors.append([abs(p - a) for p, a in zip(step_preds_original, step_actuals)])

    horizon_errors = np.array(horizon_errors)
    mean_error_per_horizon = horizon_errors.mean(axis=0)
    plt.plot(range(1, 8), mean_error_per_horizon, marker="o")
    plt.xlabel("forecast horizon (days ahead)")
    plt.ylabel("mean absolute error")
    plt.savefig(Path(__file__).parent / "horizon_error.png")
    plt.close()
    print("\nMean absolute error by forecast horizon (1-7 days ahead):", mean_error_per_horizon.round(2))


if __name__ == "__main__":
    main()
