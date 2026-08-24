# 03 — Solutions: CNN Fundamentals

## 1. Convolution from scratch

```python
import numpy as np

def conv2d_scratch(image, kernel):
    kh, kw = kernel.shape
    h, w = image.shape
    out_h, out_w = h - kh + 1, w - kw + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = image[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

image = np.zeros((6, 6))
image[:, 3:] = 1.0   # vertical edge at column 3

kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
result = conv2d_scratch(image, kernel)
print(result)
```

**Actual output** (verified by running this exact logic):

```text
[[ 0. -3. -3.  0.]
 [ 0. -3. -3.  0.]
 [ 0. -3. -3.  0.]
 [ 0. -3. -3.  0.]]
```

The two middle columns, which straddle the edge, show a strong response
(-3, since this kernel is "left minus right" and the right side is
brighter); the flat regions on either side correctly show 0 — the kernel
detects the transition, exactly as a vertical-edge detector should. The
sign is negative here only because of this kernel's specific
positive-then-negative layout — flipping the kernel horizontally would flip
the sign, not the location, of the response.

## 2. Compare to PyTorch

```python
import torch
import torch.nn.functional as F

image_t = torch.tensor(image, dtype=torch.float32).reshape(1, 1, 6, 6)
kernel_t = torch.tensor(kernel, dtype=torch.float32).reshape(1, 1, 3, 3)

torch_result = F.conv2d(image_t, kernel_t).squeeze().numpy()
print(np.allclose(result, torch_result))   # True
```

## 3. Output shape calculation

```
32x32 -> Conv2d(k=3,p=1,s=1): (32+2-3)/1+1 = 32  -> 32x32
       -> MaxPool2d(2): 32/2 = 16                 -> 16x16
       -> Conv2d(k=3,p=1,s=1): (16+2-3)/1+1 = 16   -> 16x16
       -> MaxPool2d(2): 16/2 = 8                    -> 8x8
```

```python
import torch.nn as nn

layers = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1), nn.MaxPool2d(2),
    nn.Conv2d(4, 4, 3, padding=1), nn.MaxPool2d(2),
)
dummy = torch.randn(1, 1, 32, 32)
print(layers(dummy).shape)   # torch.Size([1, 4, 8, 8]) -- matches hand calc
```

## 4. Visualizing hand-designed kernels

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
img = digits.images[0]

horizontal = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
vertical = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
blur = np.ones((3,3)) / 9

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("original")
axes[1].imshow(conv2d_scratch(img, horizontal), cmap="gray"); axes[1].set_title("horizontal edges")
axes[2].imshow(conv2d_scratch(img, vertical), cmap="gray"); axes[2].set_title("vertical edges")
axes[3].imshow(conv2d_scratch(img, blur), cmap="gray"); axes[3].set_title("blur")
plt.show()
```

## 5. Visualizing learned filters

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16*4*4, 10),
)
# ... train as in Project 006 ...

filters = model[0].weight.detach()   # shape (8, 1, 3, 3)
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(filters[i, 0], cmap="gray")
    ax.axis("off")
plt.show()
```

After training, some learned filters often resemble simple oriented edge
or blob detectors (though at this small scale with few training images,
they can also look fairly unstructured/noisy) — a real but sometimes subtle
version of the "early layers learn edges" pattern that's much more visually
striking on large, natural-image-trained CNNs.

## 6. Receptive field calculation

```
Layer 1 (3x3 conv): each output pixel sees a 3x3 region of the input.
Layer 2 (3x3 conv): each output pixel sees a 3x3 region of layer 1's output,
                     each of which itself covers a 3x3 region of the input,
                     overlapping by 1 pixel per step -> total 5x5 region.
```

General rule for stacked `k x k` convolutions (stride 1, no pooling):
receptive field grows by `(k-1)` per layer. For `k=3`: 2 layers ->
`3 + 2*(3-1) - 2 = 5`, i.e. a 5x5 receptive field — matches the direct
reasoning above. This compounding is exactly why deep networks (many
stacked layers) can recognize large-scale patterns despite each individual
filter being small — receptive field grows with depth, covering
progressively more of the original image.
