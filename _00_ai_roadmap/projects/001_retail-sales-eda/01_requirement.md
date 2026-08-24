# 01 — Requirement: Retail Sales EDA

## The brief

> "We exported two years of transactions from the store system
> (`retail_sales.csv`). Leadership wants to understand our sales performance
> before the next planning meeting. Look into it and tell us what matters."

That's it — that's the whole brief. Real stakeholders rarely hand you a clean
spec; part of the job is turning a vague ask into concrete questions.

## What to produce

1. **Data quality report** — load `retail_sales.csv` and identify: missing
   values (which columns, how many), duplicate rows, and any invalid values
   (e.g. negative amounts/quantities). Decide how to handle each, and document
   *why*.

2. **Sales trend analysis** — total revenue over time (monthly). Is there
   seasonality? Any anomalies (e.g. a month far outside the trend)?

3. **Category/city breakdown** — which product categories and which cities
   drive the most revenue? Are the top contributors stable across the two
   years, or shifting?

4. **Customer behavior** — distribution of order value and quantity per order.
   Are there a few very large orders skewing averages (use median alongside
   mean)?

5. **At least 5 charts** total across the above, each with a one- or
   two-sentence caption explaining what it shows and why it matters.

6. **A findings summary** (half a page, plain language, no code) written as if
   for a non-technical stakeholder — the actual deliverable of an EDA task,
   not the code that produced it.

## Dataset schema

| Column | Type | Notes |
|---|---|---|
| `order_id` | int | unique per order |
| `order_date` | date | `YYYY-MM-DD` |
| `city` | str | `NY`, `LA`, `SF`, `CHI` |
| `category` | str | `Electronics`, `Clothing`, `Home`, `Toys`, `Grocery` |
| `product` | str | product name within category |
| `quantity` | int | units purchased (may contain bad data) |
| `unit_price` | float | price per unit in USD |
| `total_amount` | float | `quantity * unit_price` (may contain bad data) |
| `customer_id` | int | repeats across orders |

## Constraints

- Use only what's covered in lessons 001–005 (Python, NumPy, Pandas,
  Matplotlib/Seaborn). No modeling yet — this is pure EDA.
- Don't peek at `02_solutions/` until you've produced your own report.
