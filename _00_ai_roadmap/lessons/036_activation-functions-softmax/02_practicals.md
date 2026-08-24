# 02 — Practicals: Activation Functions & Softmax

1. Implement `sigmoid`, `tanh`, `relu`, and `leaky_relu` from scratch. Plot
   all four on the same chart for `x` from -10 to 10.

2. Implement each activation's derivative (`sigmoid'`, `tanh'`, `relu'`,
   `leaky_relu'`) and plot them alongside the functions. Visually confirm
   sigmoid and tanh's derivatives approach 0 at the extremes (vanishing
   gradient) while ReLU's derivative stays exactly 1 for all positive
   inputs.

3. Simulate the vanishing gradient problem directly: chain the sigmoid
   derivative multiplicatively 10 times (`sigmoid'(x) ** 10` for some
   `x` where sigmoid isn't near its center, e.g. `x=3`) vs chaining ReLU's
   derivative 10 times for a positive `x`. Compare the magnitudes — this
   product is literally what happens to a gradient backpropagating through
   10 saturated layers (Lesson 037 covers why multiplicatively chaining is
   the backprop mechanism).

4. Implement numerically-stable `softmax` from scratch (Lesson 007's
   max-subtraction trick). Test on `z = [2.0, 1.0, 0.1]` and
   `z = [1000, 1001, 1002]` — confirm no `nan`/`inf` in either case.

5. Implement temperature-scaled softmax `softmax(z / T)`. For
   `z = [2.0, 1.0, 0.5]`, compute and plot the resulting distribution at
   `T = [0.1, 0.5, 1.0, 2.0, 5.0]` (bar chart per T). Confirm low T sharpens
   toward one-hot and high T flattens toward uniform.

6. Using `sklearn.neural_network.MLPClassifier`, train the same
   architecture on a nonlinear dataset (`make_moons`) with
   `activation="logistic"` (sigmoid) vs `activation="relu"`, using a deeper
   network (`hidden_layer_sizes=(20,20,20)`). Compare training time to
   convergence and final accuracy — does ReLU train noticeably faster or
   better on the deeper network, consistent with the vanishing gradient
   discussion?
