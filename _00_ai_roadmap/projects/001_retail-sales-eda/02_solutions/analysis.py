"""Reference EDA solution for Project 001 — Retail Sales EDA.

Run after generating the data:
    python data/generate_data.py
    python analysis.py

Produces printed data-quality/summary stats and saves charts to ./charts/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).parent / "data" / "retail_sales.csv"
CHARTS_DIR = Path(__file__).parent / "charts"
CHARTS_DIR.mkdir(exist_ok=True)


def load_raw():
    return pd.read_csv(DATA_PATH, parse_dates=["order_date"])


def data_quality_report(df):
    print("=== Data quality report ===")
    print("Rows:", len(df))
    print("\nMissing values per column:")
    print(df.isna().sum())

    dupes = df.duplicated(subset=[c for c in df.columns if c != "order_id"]).sum()
    print(f"\nDuplicate rows (ignoring order_id): {dupes}")

    negative = (df["total_amount"] < 0).sum()
    print(f"Rows with negative total_amount: {negative}")


def clean(df):
    """
    Decisions (documented, not silent):
    - Drop exact duplicate rows (same order double-exported) - keep first.
    - Missing `city`: drop the row. City is central to one of our required
      breakdowns and imputing a location would be misleading.
    - Missing `quantity`: recompute from total_amount / unit_price when both
      are present and valid, since we can recover it exactly instead of
      guessing.
    - Negative `total_amount`: these look like refunds recorded as sales.
      Drop them from the *revenue* analysis (they're not real sales) but note
      the count in the report - don't just silently discard without saying so.
    """
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "order_id"])
    df = df.dropna(subset=["city"])

    recoverable = df["quantity"].isna() & df["unit_price"].notna() & df["total_amount"].notna()
    df.loc[recoverable, "quantity"] = (
        df.loc[recoverable, "total_amount"] / df.loc[recoverable, "unit_price"]
    ).round()
    df = df.dropna(subset=["quantity"])

    n_negative = (df["total_amount"] < 0).sum()
    df = df[df["total_amount"] >= 0]

    print(f"\nDropped {n_negative} negative-amount rows from revenue analysis.")
    print(f"Rows remaining after cleaning: {len(df)}")
    return df


def sales_trend(df):
    monthly = df.set_index("order_date").resample("ME")["total_amount"].sum()

    fig, ax = plt.subplots(figsize=(9, 4))
    monthly.plot(ax=ax, marker="o")
    ax.set_title("Monthly revenue")
    ax.set_ylabel("Revenue (USD)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "monthly_revenue.png")
    plt.close(fig)

    print("\n=== Monthly revenue ===")
    print(monthly)
    return monthly


def category_city_breakdown(df):
    by_category = df.groupby("category")["total_amount"].sum().sort_values(ascending=False)
    by_city = df.groupby("city")["total_amount"].sum().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    by_category.plot(kind="bar", ax=axes[0], title="Revenue by category")
    by_city.plot(kind="bar", ax=axes[1], title="Revenue by city")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "category_city_breakdown.png")
    plt.close(fig)

    print("\n=== Revenue by category ===")
    print(by_category)
    print("\n=== Revenue by city ===")
    print(by_city)
    return by_category, by_city


def order_value_distribution(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["total_amount"], bins=40, kde=True, ax=ax)
    ax.axvline(df["total_amount"].mean(), color="red", linestyle="--", label="mean")
    ax.axvline(df["total_amount"].median(), color="green", linestyle="--", label="median")
    ax.set_title("Order value distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "order_value_distribution.png")
    plt.close(fig)

    print("\n=== Order value stats ===")
    print(df["total_amount"].describe())


def main():
    raw = load_raw()
    data_quality_report(raw)
    df = clean(raw)
    sales_trend(df)
    category_city_breakdown(df)
    order_value_distribution(df)
    print(f"\nCharts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
