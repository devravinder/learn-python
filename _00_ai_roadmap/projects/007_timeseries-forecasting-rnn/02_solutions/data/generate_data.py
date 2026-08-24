"""Generate a synthetic daily sales time series with trend + weekly
seasonality + noise. Stdlib only."""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(3)
N_DAYS = 730
START = date(2024, 1, 1)

# Monday=0 ... Sunday=6; weekend boosted
DAY_OF_WEEK_EFFECT = [0, -5, -3, 0, 5, 25, 30]


def main():
    rows = []
    for i in range(N_DAYS):
        d = START + timedelta(days=i)
        trend = 100 + i * 0.15
        seasonality = DAY_OF_WEEK_EFFECT[d.weekday()]
        noise = random.gauss(0, 8)
        sales = max(0, trend + seasonality + noise)
        rows.append({"date": d.isoformat(), "sales": round(sales, 2)})

    out_path = Path(__file__).parent / "sales.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "sales"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
