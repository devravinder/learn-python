# 01 — Concepts: CNN Architectures

## LeNet (1998) — the original

`Conv -> Pool -> Conv -> Pool -> FC -> FC -> output`, applied to digit
recognition — essentially what Project 006's CNN and Lesson 043's examples
already built. Small by modern standards (a few thousand parameters) but
established the whole conv-pool-FC pattern still used today.

## VGG (2014) — deeper, with a simple rule

VGG's insight: stack many small `3x3` convolutions instead of fewer large
ones. Two stacked 3x3 convs have the same receptive field as one 5x5 conv
(Lesson 043's receptive field math) but **fewer parameters** (`2*(3*3) = 18`
vs `5*5=25` per channel pair) **and an extra ReLU nonlinearity** in between
— strictly more expressive for the same or lower cost. VGG networks are
just this pattern repeated very deep (16-19 layers), with occasional
`MaxPool2d` to downsample.

```
[Conv3x3 -> ReLU] x2 -> MaxPool -> [Conv3x3 -> ReLU] x2 -> MaxPool -> ... -> FC layers
```

## The problem VGG-style depth eventually hits: degradation

You might expect "just keep stacking layers" to keep improving accuracy
indefinitely. In practice, very deep plain networks (past a certain depth)
get *worse* — not from overfitting (training accuracy also degrades), but
from **optimization difficulty**: gradients have to flow back through many
layers (Lesson 036's vanishing gradient problem, still relevant even with
ReLU at extreme depth), and it becomes harder for the network to even learn
an identity mapping (just "pass the input through unchanged") when that
would be the right thing to do for some layers.

## ResNet (2015) — the residual connection

The fix: instead of a block learning `H(x)` directly, have it learn the
**residual** `F(x) = H(x) - x`, and add the input back:

```
output = F(x) + x    # "skip connection" / "residual connection"
```

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
        return torch.relu(out + identity)   # the skip connection
```

Why this helps: if the ideal function for a block really is close to
identity, the network only needs to learn `F(x) ≈ 0` (easy — push weights
toward zero) rather than learning to *reconstruct* the identity mapping
through a full nonlinear transformation (hard). It also gives gradients a
direct, unimpeded path backward through the `+x` connection, bypassing
however many nonlinear layers are in `F` — directly mitigating the
vanishing gradient problem at extreme depth. ResNets successfully trained
networks over 100 layers deep, previously impractical.

## Why this matters far beyond CNNs

**The residual connection is not a CNN-only trick.** Every Transformer
block (Lesson 060) uses `output = x + Sublayer(x)` around both its
attention and feedforward components, for exactly the same reason: it lets
you stack dozens of Transformer layers (as every real LLM does) while
keeping gradients flowing cleanly during training. Understanding *why*
ResNet's skip connection works is understanding a piece of math you will
see again, unchanged in spirit, in the architecture you build your own LLM
on top of.

## Other architectural ideas worth knowing (briefly)

- **Inception/GoogLeNet**: run multiple kernel sizes in parallel within a
  block, concatenate results — lets the network choose what scale of
  pattern matters at each layer rather than committing to one kernel size.
- **Batch Normalization** (Lesson 042) is used extensively throughout
  modern CNN architectures, typically after each conv and before the
  activation.
- **Global Average Pooling**: replacing large fully-connected layers at the
  end with an average over each channel's entire spatial map — drastically
  reduces parameter count in the classifier head.

## Transfer learning (a practical payoff)

A CNN pretrained on a large dataset (e.g. ImageNet) has already learned
generally useful low/mid-level visual features (edges, textures, shapes) in
its early layers. **Transfer learning** reuses those pretrained weights and
only retrains the final layer(s) for a new, smaller task — dramatically
reducing the data and compute needed compared to training from scratch.
This exact idea — reuse a large pretrained model, fine-tune a small part
for your task — is also the core idea behind Lesson 069's LLM fine-tuning,
just applied to language models instead of CNNs.
