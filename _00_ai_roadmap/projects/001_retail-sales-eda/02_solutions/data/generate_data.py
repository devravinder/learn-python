"""Generate a synthetic, intentionally-messy retail sales dataset.

Stdlib only (no numpy/pandas) so it can run before you've even set up your
environment (Lesson 001). Deterministic via a fixed random seed.

Usage:
    python generate_data.py
    -> writes retail_sales.csv next to this script
"""
import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(42)

CITIES = ["NY", "LA", "SF", "CHI"]
CATEGORIES = {
    "Electronics": [("Headphones", 40), ("Laptop", 900), ("Phone Case", 15)],
    "Clothing": [("T-Shirt", 20), ("Jeans", 55), ("Jacket", 120)],
    "Home": [("Blender", 45), ("Lamp", 30), ("Cookware Set", 85)],
    "Toys": [("Board Game", 25), ("Action Figure", 12), ("Puzzle", 18)],
    "Grocery": [("Coffee Beans", 14), ("Olive Oil", 22), ("Snack Box", 10)],
}
N_CUSTOMERS = 500
START = date(2024, 1, 1)
END = date(2025, 12, 31)
N_ROWS = 3000


def random_date():
    span = (END - START).days
    d = START + timedelta(days=random.randint(0, span))
    # holiday-season seasonality: bias sampling toward Nov/Dec by resampling
    if d.month not in (11, 12) and random.random() < 0.35:
        year = d.year
        d = date(year, random.choice([11, 12]), random.randint(1, 28))
    return d


def main():
    rows = []
    for order_id in range(1, N_ROWS + 1):
        category = random.choice(list(CATEGORIES.keys()))
        product, base_price = random.choice(CATEGORIES[category])
        unit_price = round(base_price * random.uniform(0.85, 1.15), 2)
        quantity = random.choices([1, 2, 3, 4, 5], weights=[40, 30, 15, 10, 5])[0]
        total_amount = round(unit_price * quantity, 2)

        row = {
            "order_id": order_id,
            "order_date": random_date().isoformat(),
            "city": random.choice(CITIES),
            "category": category,
            "product": product,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_amount": total_amount,
            "customer_id": random.randint(1, N_CUSTOMERS),
        }
        rows.append(row)

    # --- inject realistic messiness ---

    # 1. missing quantity (~1.5%)
    for row in random.sample(rows, int(N_ROWS * 0.015)):
        row["quantity"] = ""

    # 2. missing city (~1%)
    for row in random.sample(rows, int(N_ROWS * 0.01)):
        row["city"] = ""

    # 3. a handful of negative amounts (refunds recorded incorrectly, ~0.5%)
    for row in random.sample(rows, int(N_ROWS * 0.005)):
        row["total_amount"] = -abs(row["total_amount"])

    # 4. duplicate rows (~1%) - same order double-exported
    dupes = [dict(r) for r in random.sample(rows, int(N_ROWS * 0.01))]
    rows.extend(dupes)

    random.shuffle(rows)

    out_path = Path(__file__).parent / "retail_sales.csv"
    fieldnames = [
        "order_id", "order_date", "city", "category", "product",
        "quantity", "unit_price", "total_amount", "customer_id",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
