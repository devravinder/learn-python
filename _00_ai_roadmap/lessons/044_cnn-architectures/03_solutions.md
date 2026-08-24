# 03 — Solutions: CNN Architectures

## 1. Parameter count: two 3x3 vs one 5x5

```python
import torch
import torch.nn as nn

vgg_style = nn.Sequential(
    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
)
single_5x5 = nn.Sequential(
    nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(),
)

vgg_params = sum(p.numel() for p in vgg_style.parameters())
single_params = sum(p.numel() for p in single_5x5.parameters())
print("two 3x3 convs:", vgg_params)     # 2 * (16*16*3*3 + 16) = 4640
print("one 5x5 conv:", single_params)    # 16*16*5*5 + 16 = 6416
```

Two stacked 3x3 convs (4,640 params) use fewer parameters than one 5x5 conv
(6,416 params) while covering the same 5x5 receptive field and adding an
extra nonlinearity — exactly VGG's efficiency argument from
`01_concepts.md`.

## 2. ResidualBlock shape check

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + identity)

block = ResidualBlock(16)
x = torch.randn(4, 16, 8, 8)
print(block(x).shape)   # torch.Size([4, 16, 8, 8]) -- unchanged, as required
```

## 3. Plain deep vs residual on digits

```python
plain_deep = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
    *[layer for _ in range(5) for layer in (nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())],
    nn.Flatten(), nn.Linear(16*8*8, 10),
)

class ResNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 16, 3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(16) for _ in range(3)])
        self.head = nn.Linear(16*8*8, 10)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        x = self.blocks(x)
        return self.head(x.flatten(1))

# train both with the Project 006 workflow and compare loss curves
```

At only 6-7 layers deep, the plain network likely still trains fine on this
small/easy dataset — ResNet's original degradation problem was demonstrated
at 20+ and 50+ layers; at this shallow depth you may see little to no
difference, which is itself a useful, honest finding: skip connections
solve a *specific* problem (very deep network optimization) that doesn't
necessarily show up at shallow depth. Try 20+ stacked plain conv layers to
more reliably reproduce degradation.

## 4. Breaking the skip connection

```python
class BrokenResidualBlock(ResidualBlock):
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out)   # skip connection removed
```

On a shallow network like this small example, the difference may again be
subtle; the effect becomes much more pronounced as depth increases — the
whole point of the original ResNet paper was demonstrating this gap
specifically at extreme depth (comparing 34-layer plain vs 34-layer
residual networks, where the plain version performed *worse* than an
18-layer plain network — a direct optimization difficulty, not overfitting).

## 5. Transfer learning with a pretrained ResNet

```python
import torchvision.models as models

model = models.resnet18(weights="IMAGENET1K_V1")
for param in model.parameters():
    param.requires_grad = False   # freeze all pretrained layers

model.fc = nn.Linear(model.fc.in_features, 10)   # new trainable head

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)   # only trains the new layer
```

With frozen pretrained features, expect reasonable accuracy after
surprisingly few epochs and little data compared to training a CNN from
scratch — the pretrained early/mid layers already encode generally useful
visual features that transfer to most natural-image tasks.

## 6. VGG16 vs ResNet18 parameter count

```python
vgg16 = models.vgg16()
resnet18 = models.resnet18()

vgg_total = sum(p.numel() for p in vgg16.parameters())
resnet_total = sum(p.numel() for p in resnet18.parameters())
print("VGG16:", vgg_total)        # ~138 million
print("ResNet18:", resnet_total)  # ~11 million
```

VGG16 has roughly 12x more parameters than ResNet18, mostly concentrated in
VGG's large fully-connected classifier layers at the end (a design later
architectures moved away from, favoring global average pooling instead) —
ResNet18 achieves comparable or better ImageNet accuracy with far fewer
parameters largely due to (a) global average pooling instead of large FC
layers and (b) residual connections enabling effective use of depth without
needing extremely wide layers to compensate for optimization difficulty.
