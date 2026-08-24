"""Generate a synthetic-but-realistic housing dataset. Stdlib only.

Usage:
    python generate_data.py
    -> writes housing.csv next to this script
"""
import csv
import random
from pathlib import Path

random.seed(7)

N = 500


def main():
    rows = []
    for i in range(1, N + 1):
        sqft = random.uniform(500, 4000)
        bedrooms = random.randint(1, 6)
        age = random.uniform(0, 50)
        distance = random.uniform(0, 30)

        price = (
            50_000
            + 150 * sqft
            + 10_000 * bedrooms
            - 800 * age
            - 2_000 * distance
            + random.gauss(0, 15_000)
        )
        price = max(price, 20_000)  # floor, avoid nonsensical negative prices

        rows.append({
            "sqft": round(sqft, 1),
            "bedrooms": bedrooms,
            "age": round(age, 1),
            "distance_km": round(distance, 2),
            "price": round(price, 2),
        })

    out_path = Path(__file__).parent / "housing.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sqft", "bedrooms", "age", "distance_km", "price"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
