# Findings — Image Classifier (Intro CNN)

*(Caveat, unlike other projects' FINDINGS in this repo: the sandbox this
curriculum was authored in has no PyTorch installed, so — unlike Projects
001–005, where numbers were independently verified with runnable stdlib/
pure-Python checks — the figures below are **well-established expected
ranges** for this exact dataset/architecture combination, not numbers
produced by actually executing `analysis.py` here. Run it yourself and
replace this note with your real numbers.)*

## Expected results

On `sklearn`'s digits dataset (1797 samples, 8x8 images, 10 classes), both
a plain MLP and a small CNN with this project's architectures typically
reach **95-99% test accuracy** — this dataset is small, clean, and the
digits are centered/normalized, making it comparatively easy for either
architecture.

## Why the CNN vs MLP gap is expected to be modest here

Unlike harder image tasks, 8x8 digit images have very little spatial
complexity for a CNN's translation-invariance and local-pattern-detection
advantages (Lesson 043 covers *why* these exist) to meaningfully
outperform a plain MLP that just treats pixels as an unordered feature
vector. Expect the CNN to match or modestly beat the MLP (a percentage
point or two), **not** a dramatic gap — that gap becomes much clearer on
harder, larger, more varied real-world images (a good empirical preview of
why Module 7 exists).

## Dropout on a small, clean dataset

Since this dataset is small (only ~1440 training images) but also fairly
low-noise and low-ambiguity, adding dropout to the MLP may provide little
benefit or even slightly hurt training accuracy within a fixed epoch budget
— a useful real lesson from Lesson 042: regularization isn't free, and its
benefit depends on whether overfitting is actually a problem for your
specific data/model combination. Check your own train/test gap before
assuming dropout should help.

## Error analysis

Expect most misclassifications to involve visually similar digit pairs
(commonly 3/5, 4/9, 7/1 at low resolution) — genuinely ambiguous even to a
human glancing quickly at an 8x8 grayscale rendering, rather than clear
model failures. If you see errors that look obviously easy to a human, that
usually points to a preprocessing bug (e.g. forgetting to normalize pixel
values, or a train/test data leak) rather than a genuine model limitation.
