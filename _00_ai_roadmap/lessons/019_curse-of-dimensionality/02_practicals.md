# 02 — Practicals: Curse of Dimensionality

1. Using the `distance_ratio` function from `01_concepts.md`, compute the
   nearest/farthest distance ratio for dimensions `[1, 2, 5, 10, 50, 100,
   500]` with 1000 random points each. Plot ratio vs dimension. At what
   dimension does the ratio already exceed 0.9?

2. Generate 1000 random points uniformly in a `d`-dimensional unit cube for
   `d = [2, 10, 50]`. For each `d`, compute what fraction of points fall
   within the inscribed unit sphere (distance from center < 0.5). Watch the
   fraction shrink toward 0 as `d` grows — this is the same "volume
   concentrates near the corners, not the center" phenomenon in a different
   guise.

3. **KNN breaking down empirically**: generate a simple 2-class dataset in
   2D that's linearly separable, train/evaluate a from-scratch KNN classifier
   (k=5, Euclidean distance) via cross-validation accuracy. Now add 200
   random noise dimensions (irrelevant features) to every point and re-run
   the same KNN. How much does accuracy drop, and why?

4. Using the dataset from Q3 (2D + noise dimensions), apply PCA (you can use
   `sklearn.decomposition.PCA` here — full from-scratch PCA is Lesson 031) to
   reduce back down to 2 dimensions before running KNN. Does accuracy
   recover close to the original 2D-only performance?

5. Explain in your own words why "collect every feature you can think of and
   let the model figure out what matters" is not free, referencing both the
   overfitting risk and the distance-concentration effect from Q1–Q2.
