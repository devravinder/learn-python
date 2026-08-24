# 01 — Concepts: CNN Fundamentals

## Why not just flatten the image and use an MLP?

You did exactly that in Project 006, and it worked reasonably on tiny 8x8
digits. But flattening throws away spatial structure — pixel `(3,4)` being
near pixel `(3,5)` is meaningful information an MLP never sees, since
flattening treats every pixel as an independent, unordered feature. It also
doesn't scale: a 224x224 RGB image flattened is 150,528 features — a fully
connected layer from that to even 1000 hidden units is 150 million weights,
in the *first* layer alone.

## The convolution operation

A small learnable **filter/kernel** (e.g. 3x3) slides across the image,
computing a dot product (Lesson 010) at each position:

```
output[i,j] = Σ_(di,dj) input[i+di, j+dj] * kernel[di, dj]
```

This is the same dot-product-based similarity idea from Lesson 010, applied
locally and repeated across the whole image using the **same** kernel
weights at every position.

## Why convolutions are well-suited to images: two key properties

1. **Local connectivity**: each output value depends only on a small
   neighborhood of the input, not the entire image — matches the reality
   that meaningful visual patterns (edges, textures) are local.
2. **Parameter sharing (translation invariance)**: the *same* kernel is
   applied everywhere in the image, so a kernel that learns to detect
   "vertical edge" detects it wherever it appears — a cat in the top-left
   or bottom-right of an image is recognized by the same learned features.
   This is dramatically more parameter-efficient than a fully connected
   layer, and generalizes better to shifted/translated versions of a
   pattern.

## What early filters learn

In a trained CNN, the first convolutional layer typically learns simple
patterns — edge detectors at various orientations, color blobs. Deeper
layers combine these into increasingly complex patterns (corners, textures,
then parts, then whole objects) — a hierarchical feature-learning structure
that (loosely) echoes how biological visual cortex processes information in
stages.

## Key hyperparameters

- **Kernel size**: typically 3x3 or 5x5 — how large a local neighborhood
  each filter looks at.
- **Stride**: how far the kernel moves each step; stride 2 downsamples the
  output by roughly half.
- **Padding**: adding zeros around the input border so the output can stay
  the same size as the input (`padding="same"` in modern APIs) — otherwise
  the output shrinks slightly with every conv layer (an unpadded 3x3 kernel
  on an `NxN` image gives `(N-2)x(N-2)` output).
- **Number of filters (output channels)**: each filter learns a different
  pattern; more filters = more different features detected per layer, at
  the cost of more parameters/compute.

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
```

## Pooling: downsampling while keeping the strongest signal

**Max pooling** takes the maximum value in each small window (e.g. 2x2),
reducing spatial resolution while keeping the most salient activation —
provides a small amount of translation invariance (a feature detected
slightly shifted still triggers the same max) and reduces computation for
deeper layers.

```python
pool = nn.MaxPool2d(kernel_size=2)   # halves height and width
```

## Computing output shape (you'll need this constantly)

```
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
```

For a `28x28` input, `kernel_size=3, padding=1, stride=1`:
`(28 + 2 - 3)/1 + 1 = 28` — same size (this is exactly what "same" padding
means for a 3x3 kernel). After a `MaxPool2d(2)`: `28 -> 14`.

## Receptive field: how much of the input each unit "sees"

A unit deep in the network is influenced by a progressively larger region
of the original input as you stack more conv/pool layers — its
**receptive field** grows with depth. This is why deep CNNs can recognize
large, complex patterns (whole objects) despite each individual filter only
looking at a small local window directly.

## Putting it together: a typical CNN "block"

```
Conv -> (BatchNorm, Lesson 042) -> ReLU -> [repeat] -> Pool -> [repeat blocks] -> Flatten -> FC layers -> output
```

Exactly the structure you already built in Project 006 — this lesson gives
you the vocabulary and reasoning for *why* that structure, setting up
Lesson 044's tour of real architectures (LeNet, VGG, ResNet) built from
these same blocks at increasing scale and sophistication.
