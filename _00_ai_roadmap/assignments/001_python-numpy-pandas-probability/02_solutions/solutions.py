"""Reference solutions for Assignment 001.

Run with: python solutions.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(123)
n = 500

df = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "hours_studied": rng.uniform(0, 10, n).round(1),
    "attendance_pct": rng.uniform(50, 100, n).round(1),
    "passed_prior_course": rng.choice([True, False], n, p=[0.7, 0.3]),
})

df["exam_score"] = (
    30
    + 4.5 * df["hours_studied"]
    + 0.2 * df["attendance_pct"]
    + 5 * df["passed_prior_course"]
    + rng.normal(0, 6, n)
).clip(0, 100).round(1)

missing_idx = rng.choice(n, size=15, replace=False)
df.loc[missing_idx, "attendance_pct"] = np.nan


# --- Q1: letter_grade ---
def letter_grade(score):
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


df["grade"] = df["exam_score"].apply(letter_grade)
print("Q1 grade counts:\n", df["grade"].value_counts(), "\n")


# --- Q2: mean without .mean() ---
scores_np = df["exam_score"].to_numpy()
manual_mean = scores_np.sum() / len(scores_np)
assert np.isclose(manual_mean, scores_np.mean())
print("Q2 manual mean:", manual_mean, "== .mean():", scores_np.mean(), "\n")


# --- Q3: within one std ---
mean, std = scores_np.mean(), scores_np.std()
within_1std = ((scores_np > mean - std) & (scores_np < mean + std)).sum()
print(f"Q3 scores within 1 std of mean: {within_1std} / {len(scores_np)}\n")


# --- Q4: fill missing attendance ---
# Median is preferred here because attendance_pct could plausibly have a
# skewed distribution (a cluster of near-100% attendees), and median is
# robust to that skew whereas the mean would be pulled toward it.
df["attendance_pct"] = df["attendance_pct"].fillna(df["attendance_pct"].median())
print("Q4 remaining missing attendance:", df["attendance_pct"].isna().sum(), "\n")


# --- Q5: groupby prior course ---
by_prior = df.groupby("passed_prior_course")["exam_score"].mean()
print("Q5 avg exam_score by passed_prior_course:\n", by_prior, "\n")
# Expected: True group scores higher, since the data was generated with
# `+ 5 * passed_prior_course` added directly to exam_score.


# --- Q6: study buckets ---
df["study_bucket"] = pd.cut(
    df["hours_studied"],
    bins=[-0.01, 3.3, 6.6, 10.0],
    labels=["low", "medium", "high"],
)
print("Q6 avg exam_score by study_bucket:\n", df.groupby("study_bucket")["exam_score"].mean(), "\n")
# Expected: monotonically increasing low -> medium -> high, since exam_score
# was generated with a positive linear term on hours_studied.


# --- Q7: scatter plot ---
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="hours_studied", y="exam_score", hue="passed_prior_course", ax=ax)
fig.savefig("q7_scatter.png")
plt.close(fig)


# --- Q8: correlation heatmap ---
fig, ax = plt.subplots()
corr = df[["hours_studied", "attendance_pct", "exam_score"]].corr()
sns.heatmap(corr, annot=True, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)
fig.savefig("q8_heatmap.png")
plt.close(fig)


# --- Q9: conditional probabilities from data ---
p_passed = df["passed_prior_course"].mean()
p_a_given_passed = (df.loc[df["passed_prior_course"], "grade"] == "A").mean()
p_a_given_not_passed = (df.loc[~df["passed_prior_course"], "grade"] == "A").mean()
print("Q9 P(passed_prior_course):", p_passed)
print("Q9 P(A | passed):", p_a_given_passed)
print("Q9 P(A | not passed):", p_a_given_not_passed, "\n")
# Expected: P(A | passed) > P(A | not passed), consistent with the
# +5 point boost baked into the data generator.


# --- Q10: correlation + z-score of top scorer ---
corr_hs_score = df["hours_studied"].corr(df["exam_score"])
top_student = df.loc[df["exam_score"].idxmax()]
z_hours = (top_student["hours_studied"] - df["hours_studied"].mean()) / df["hours_studied"].std()
print("Q10 corr(hours_studied, exam_score):", corr_hs_score)
print("Q10 top student:\n", top_student)
print("Q10 z-score of top student's hours_studied:", z_hours)
# If z_hours is only modestly positive (e.g. < 1), the top score is likely
# explained more by a combination of attendance/prior-course/noise than by
# hours studied alone -- worth checking top_student's attendance_pct and
# passed_prior_course values directly, not just hours_studied.
