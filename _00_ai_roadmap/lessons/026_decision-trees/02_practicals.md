# 02 — Practicals: Decision Trees

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=0)
```

1. Fit `DecisionTreeClassifier()` with no depth limit. Report train and test
   accuracy (80/20 split). Is there a large gap suggesting overfitting?

2. Sweep `max_depth` from 1 to 15 and plot train/test accuracy vs depth
   (same style as Lesson 025's `k` sweep). Identify the best depth.

3. Implement Gini impurity from scratch: `gini(labels) = 1 - sum(p_i^2)`.
   Compute it for a fully mixed node (`[0,0,1,1]`) and a pure node
   (`[1,1,1,1]`) — confirm pure gives 0 and mixed gives the maximum possible
   value for 2 classes (0.5).

4. Implement information gain from scratch for a single binary split: given
   a parent set of labels and two child label sets after a split, compute
   `parent_entropy - weighted_average(child_entropies)`. Test it on a toy
   3-feature dataset where one feature perfectly separates the classes —
   confirm splitting on that feature gives higher information gain than
   splitting on a random feature.

5. Fit a tree with `max_depth=3` on the Titanic-style data from Assignment
   002 (or regenerate it) and visualize it with `plot_tree`. Read off the
   first split — does it match what you found via logistic regression
   coefficients to be the most important feature?

6. Print `model.feature_importances_` for a tree fit on the full dataset and
   compare the ranking to what a correlation-with-target analysis
   (`np.corrcoef` per feature) would suggest — do they roughly agree?
