"""Synthetic Titanic-style passenger survival dataset. Stdlib only."""
import csv
import random
from pathlib import Path

random.seed(11)
N = 600


def main():
    rows = []
    for pid in range(1, N + 1):
        pclass = random.choices([1, 2, 3], weights=[0.25, 0.25, 0.5])[0]
        sex = random.choices(["female", "male"], weights=[0.45, 0.55])[0]
        age = max(0.5, random.gauss(29, 14))
        fare = max(5.0, random.gauss({1: 80, 2: 25, 3: 12}[pclass], 15))
        sibsp = random.choices([0, 1, 2, 3], weights=[0.6, 0.25, 0.1, 0.05])[0]

        # survival probability driven by class, sex, age (women/children/1st class favored)
        logit = (
            -1.0
            + (1.8 if sex == "female" else 0.0)
            + {1: 1.5, 2: 0.5, 3: 0.0}[pclass]
            + (0.8 if age < 12 else 0.0)
            - 0.01 * age
        )
        prob = 1 / (1 + pow(2.71828, -logit))
        survived = 1 if random.random() < prob else 0

        rows.append({
            "passenger_id": pid,
            "pclass": pclass,
            "sex": sex,
            "age": round(age, 1),
            "fare": round(fare, 2),
            "sibsp": sibsp,
            "survived": survived,
        })

    # inject missing ages (~15%), a common real Titanic-dataset quirk
    for r in random.sample(rows, int(N * 0.15)):
        r["age"] = ""

    out_path = Path(__file__).parent / "titanic_synthetic.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
