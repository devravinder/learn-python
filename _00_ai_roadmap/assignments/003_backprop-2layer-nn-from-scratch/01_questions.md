# 01 — Questions

Dataset: `sklearn.datasets.make_moons(n_samples=300, noise=0.2, random_state=0)`,
labels reshaped to `(300, 1)`.

Network: 2 inputs -> 8 hidden units (ReLU) -> 1 output (sigmoid), binary
cross-entropy loss (Lessons 016, 023).

1. Implement the forward pass with NumPy matrix operations:
   ```
   z1 = X @ W1 + b1        # (n, 8)
   a1 = relu(z1)            # (n, 8)
   z2 = a1 @ W2 + b2        # (n, 1)
   y_hat = sigmoid(z2)      # (n, 1)
   ```
   Initialize `W1, W2` with small random values (e.g.
   `np.random.default_rng(0).normal(0, 0.5, size=...)`) and biases at zero.

2. Derive and implement the backward pass. Starting from binary
   cross-entropy loss, show (in a comment) that `∂L/∂z2 = y_hat - y`
   (the same clean simplification as Lesson 023's logistic regression
   gradient), then implement:
   ```
   dW2 = a1.T @ dz2
   db2 = sum(dz2, axis=0)
   da1 = dz2 @ W2.T
   dz1 = da1 * relu'(z1)
   dW1 = X.T @ dz1
   db1 = sum(dz1, axis=0)
   ```
   (Don't forget to average — or sum and then scale the learning rate — over
   the batch size `n` consistently between the loss and the gradients.)

3. Write the training loop (plain gradient descent, Lesson 015 — no
   optimizer library) and train for 2000 epochs. Plot the loss curve.

4. **Gradient check**: for `W1[0,0]` and `b2[0]`, perturb by `h=1e-5`,
   recompute the loss, and compare the numerical gradient to your
   analytical `dW1[0,0]`/`db2[0]`. Confirm they match within `1e-4`.

5. Report final training accuracy (threshold `y_hat` at 0.5). Plot the
   learned decision boundary over the moons data (evaluate the model on a
   grid of points and contour-plot `y_hat`).

6. Increase the hidden layer to 2 units instead of 8, retrain, and compare
   both the loss curve and the decision boundary plot. Does capacity
   visibly affect how well it can separate the two moons (connect to
   Lesson 017's bias-variance framing)?
