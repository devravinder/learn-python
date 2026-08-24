# 01 — Concepts: Multiple Linear Regression & Assumptions

## Multicollinearity

When two or more features are highly correlated with each other, the model
can't reliably tell them apart — small changes in the data can swing
coefficient estimates wildly, and "holding other features constant" stops
making sense (Lesson 020) because those correlated features rarely vary
independently in your actual data.

**Variance Inflation Factor (VIF)** detects this: regress each feature
against all the others; a high VIF (commonly >10 is flagged) for a feature
means it's largely predictable from the rest, i.e. redundant.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

Fixes: drop one of the correlated features, combine them (e.g. via PCA,
Lesson 031), or use regularization (Ridge, Lesson 022) which handles
correlated features more gracefully than plain linear regression.

## Homoscedasticity check

Beyond eyeballing the residual plot (Lesson 020), the **Breusch-Pagan test**
formally tests whether residual variance depends on the predicted value
(`statsmodels.stats.diagnostic.het_breuschpagan`). A significant result
(Lesson 009's hypothesis testing) suggests heteroscedasticity.

## Normality of residuals

Plot a histogram or Q-Q plot of residuals; a formal Shapiro-Wilk test
(`scipy.stats.shapiro`) checks normality. This mostly affects the validity
of confidence intervals and significance tests on coefficients — pure
prediction accuracy is fairly robust to non-normal residuals.

## Outliers and influence

Not all outliers matter equally: an outlier in `y` far from the regression
line but near the center of the `X` distribution has limited **leverage**;
an outlier extreme in `X` can single-handedly rotate the whole fitted line
(**high leverage + high influence**). **Cook's distance** quantifies how
much removing a single point would change the fitted coefficients — a
standard tool for flagging points worth investigating individually rather
than blindly dropping.

## Categorical features in linear regression

Linear regression needs numbers, not categories. **One-hot encoding**
converts a categorical column into multiple 0/1 indicator columns (drop one
category as the reference level, to avoid perfect multicollinearity between
the new columns — the "dummy variable trap"):

```python
df_encoded = pd.get_dummies(df, columns=["city"], drop_first=True)
```

## Interaction terms

Sometimes two features' *combined* effect isn't just the sum of their
individual effects (e.g. the effect of `hours_studied` on `exam_score` might
be stronger for students who also have high `attendance`). An interaction
term `x1 * x2` added as an explicit feature lets linear regression capture
this — without it, linear regression can only add effects, never let one
feature's effect depend on another's value.
