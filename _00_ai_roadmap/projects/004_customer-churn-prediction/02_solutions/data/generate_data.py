"""Generate a synthetic telecom-style churn dataset. Stdlib only."""
import csv
import random
from pathlib import Path

random.seed(21)
N = 2000


def sigmoid(x):
    return 1 / (1 + pow(2.71828, -x))


def main():
    rows = []
    for cid in range(1, N + 1):
        tenure = max(1, int(random.gauss(24, 18)))
        contract = random.choices(
            ["month-to-month", "one-year", "two-year"], weights=[0.55, 0.25, 0.2]
        )[0]
        monthly_charges = max(15, random.gauss(70, 25))
        total_charges = round(monthly_charges * tenure * random.uniform(0.9, 1.05), 2)
        support_calls = random.choices([0, 1, 2, 3, 4, 5], weights=[30, 25, 20, 12, 8, 5])[0]
        payment_method = random.choices(
            ["credit_card", "bank_transfer", "electronic_check"], weights=[0.4, 0.35, 0.25]
        )[0]
        is_senior = 1 if random.random() < 0.16 else 0
        has_dependents = 1 if random.random() < 0.3 else 0

        logit = (
            -2.0
            + (1.3 if contract == "month-to-month" else (-0.3 if contract == "two-year" else 0.0))
            - 0.03 * tenure
            + 0.35 * support_calls
            + (0.5 if payment_method == "electronic_check" else 0.0)
            + 0.01 * (monthly_charges - 70)
            - 0.3 * has_dependents
        )
        prob = sigmoid(logit)
        churned = 1 if random.random() < prob else 0

        rows.append({
            "customer_id": cid,
            "tenure_months": tenure,
            "contract_type": contract,
            "monthly_charges": round(monthly_charges, 2),
            "total_charges": total_charges,
            "support_calls": support_calls,
            "payment_method": payment_method,
            "is_senior": is_senior,
            "has_dependents": has_dependents,
            "churned": churned,
        })

    out_path = Path(__file__).parent / "churn.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    churn_rate = sum(r["churned"] for r in rows) / len(rows)
    print(f"Wrote {len(rows)} rows to {out_path}, churn rate={churn_rate:.3f}")


if __name__ == "__main__":
    main()
