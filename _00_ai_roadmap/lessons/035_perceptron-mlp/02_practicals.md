# 02 — Practicals: Perceptron & Multi-Layer Perceptron

1. Implement the classic perceptron learning rule from scratch. Train it on
   a linearly separable 2D dataset (`sklearn.datasets.make_blobs` with
   `centers=2`). Plot the decision boundary after training.

2. Try training the same perceptron on the XOR dataset:
   ```python
   X = np.array([[0,0],[0,1],[1,0],[1,1]])
   y = np.array([0,1,1,0])
   ```
   Confirm it fails to converge to a correct solution (track accuracy over
   many epochs — it should plateau well below 100%).

3. Implement a minimal 2-layer MLP forward pass from scratch (2 inputs -> 2
   hidden units with ReLU -> 1 output with sigmoid) with **hand-picked**
   weights (not trained) that correctly solve XOR. (Hint: one hidden unit
   can detect "at least one is 1," another "both are 1"; combine them.)
   Verify it gets all 4 XOR cases right.

4. Using `sklearn.neural_network.MLPClassifier` (a full trainable
   implementation, before you build your own in Lesson 038), fit an MLP
   with `hidden_layer_sizes=(4,)` on the XOR data. Confirm it learns to
   solve XOR correctly, unlike the single perceptron in Q2.

5. Vary `hidden_layer_sizes` for the `MLPClassifier` on XOR:
   `(1,)`, `(2,)`, `(4,)`, `(10,)`. Does a single hidden unit ever succeed?
   How many are actually needed at minimum, empirically?

6. Remove the nonlinear activation entirely
   (`MLPClassifier(hidden_layer_sizes=(10,10), activation="identity")`) and
   try XOR again. Confirm it fails, directly demonstrating the "stacked
   linear layers collapse into one linear layer" point from
   `01_concepts.md`.
