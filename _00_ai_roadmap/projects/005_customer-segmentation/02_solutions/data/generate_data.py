"""Generate synthetic RFM customer data from 4 latent segments. Stdlib only."""
import csv
import random
from pathlib import Path

random.seed(5)

# (recency_mean, frequency_mean, monetary_mean, basket_mean, weight)
SEGMENTS = [
    ("champions",        10,  18, 1400, 4.5, 0.20),
    ("at_risk",         180,   4,  600, 2.0, 0.25),
    ("new_customers",    20,   2,  150, 1.5, 0.30),
    ("occasional_buyers", 90,   7,  350, 2.5, 0.25),
]
N = 800


def clip(v, lo):
    return max(lo, v)


def main():
    rows = []
    weights = [s[4] for s in SEGMENTS]
    for cid in range(1, N + 1):
        name, rec_m, freq_m, mon_m, bask_m, _ = random.choices(SEGMENTS, weights=weights)[0]
        recency = clip(int(random.gauss(rec_m, rec_m * 0.3 + 5)), 0)
        frequency = clip(int(random.gauss(freq_m, freq_m * 0.35 + 1)), 0)
        monetary = round(clip(random.gauss(mon_m, mon_m * 0.3 + 20), 10), 2)
        basket = round(clip(random.gauss(bask_m, 0.6), 1), 2)
        rows.append({
            "customer_id": cid,
            "recency_days": recency,
            "frequency": frequency,
            "monetary": monetary,
            "avg_basket_size": basket,
        })

    out_path = Path(__file__).parent / "customers.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
