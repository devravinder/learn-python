# 02 — Practicals: Vectors & Vector Spaces

1. Given `a = np.array([3, -4, 12])`, compute its L1 norm, L2 norm, and unit
   vector. Confirm the unit vector has L2 norm ≈ 1.

2. Implement `cosine_similarity(a, b)` from scratch (no `scipy`/`sklearn`).
   Test on three "word embeddings" (toy 3D vectors):
   ```python
   king  = np.array([0.9, 0.1, 0.2])
   queen = np.array([0.85, 0.15, 0.25])
   apple = np.array([0.1, 0.9, 0.05])
   ```
   Confirm `king`/`queen` are more similar than `king`/`apple`.

3. Given `a = np.array([2, 0])` and `b = np.array([1, 1])`, compute the
   projection of `a` onto `b`. Verify geometrically that `a - proj_b(a)` is
   orthogonal to `b` (dot product ≈ 0).

4. A recommendation-system toy problem: represent 4 users as vectors of
   ratings across 3 movies (rows), find via cosine similarity which pair of
   users has the most similar taste.
   ```python
   users = np.array([
       [5, 4, 1],
       [4, 5, 1],
       [1, 1, 5],
       [2, 1, 4],
   ])
   ```

5. Explain why cosine similarity, not Euclidean distance, is usually
   preferred for comparing text embeddings — think about what happens to
   Euclidean distance if one embedding vector is scaled by 10x but points in
   the same direction.

6. Implement min-max normalization of a set of vectors to unit vectors
   (L2-normalize each row of a matrix) using only vectorized NumPy (no
   Python loop over rows).
