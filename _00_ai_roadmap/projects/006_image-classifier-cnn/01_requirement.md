# 01 — Requirement: Image Classifier (Intro CNN)

## The brief

Classify handwritten digits (0-9) from 8x8 grayscale images. Build two
models and compare them: a plain MLP (treating the image as a flat vector,
throwing away spatial structure) and a small CNN (respecting the 2D
structure via convolutions).

## What to produce

1. **Data prep**: load `sklearn.datasets.load_digits()`, normalize pixel
   values to `[0, 1]`, split train/val/test (stratified, Lesson 024).
   Reshape appropriately for each model (MLP wants a flat `(n, 64)` vector,
   CNN wants `(n, 1, 8, 8)` — channel dimension first, PyTorch convention).

2. **MLP baseline**: build and train an MLP (Lesson 040's workflow:
   `Dataset`/`DataLoader`, train/val loop, `nn.CrossEntropyLoss`). Report
   test accuracy and a confusion matrix (Lesson 024).

3. **Small CNN**: build a CNN with 1-2 `nn.Conv2d` layers (+ ReLU + 
   `nn.MaxPool2d`), followed by a small fully-connected head. You don't need
   to fully understand *why* convolutions work yet (that's Lesson 043) —
   just get one training. A reasonable starting point:
   ```python
   nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
   nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
   nn.Flatten(),
   nn.Linear(16 * 4 * 4, 10)
   ```
   Train with the same workflow as the MLP. Report test accuracy and
   confusion matrix.

4. **Compare**: which model gets higher test accuracy? Is the difference
   large or small on this particular (fairly easy) dataset — 8x8 digits are
   much simpler than real-world images, so don't be surprised if the gap is
   modest here (Module 7 will make the case for CNNs more dramatically on
   harder image data).

5. **Regularization check**: add `nn.Dropout` to the MLP (Lesson 042) and
   see if it changes test accuracy. Since this dataset is small and fairly
   clean, does regularization help, hurt, or make little difference?

6. **Error analysis**: plot 10 misclassified test images (for your best
   model) with their true and predicted labels. Do the errors look like
   genuinely ambiguous handwriting, or like clear model mistakes?

## Constraints

- PyTorch only (no `sklearn.neural_network`) — this project is about
  practicing the Module 6 PyTorch workflow end to end.
- Don't peek at `02_solutions/` before you have both models trained and
  compared yourself.
