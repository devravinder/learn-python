# 02 — Practicals: CNN Architectures

1. Build a "VGG-style" small block: two stacked `Conv2d(kernel_size=3, padding=1)`
   + ReLU, followed by `MaxPool2d(2)`. Compute the number of parameters in
   this block (`sum(p.numel() for p in block.parameters())`) for
   `in_channels=16, out_channels=16` and compare to a single
   `Conv2d(kernel_size=5, padding=2)` + ReLU + MaxPool with the same
   in/out channels. Confirm the VGG-style block has fewer parameters
   despite a larger effective receptive field (per Lesson 043's receptive
   field math).

2. Implement the `ResidualBlock` from `01_concepts.md`. Confirm a forward
   pass preserves the input shape (residual blocks must, since you add
   `x` back directly).

3. Build two small networks on the digits dataset (Project 006's setup):
   one "plain" deep network (6 stacked conv layers, no skip connections)
   and one with 3 `ResidualBlock`s. Train both for the same number of
   epochs and compare training loss curves — does the plain deep network
   show any sign of the degradation problem (harder to optimize) on a
   network this deep, or is 6 layers too shallow to see the effect clearly
   (ResNet's original paper demonstrated it at 20+ layers)?

4. Deliberately break a residual block by removing the skip connection
   (`return torch.relu(out)` instead of `torch.relu(out + identity)`).
   Compare training speed/final loss to the real residual version on the
   same small dataset.

5. Using `torchvision.models.resnet18(weights="IMAGENET1K_V1")` (if you
   have internet access; skip if not), freeze all layers except the final
   fully-connected one (`model.fc = nn.Linear(model.fc.in_features, 10)`),
   and fine-tune only that layer on a small custom dataset (e.g. your own
   collected images, or CIFAR-10 subset). Report how few epochs/how little
   data it takes to get reasonable accuracy — the practical payoff of
   transfer learning.

6. Count total parameters in `torchvision.models.vgg16()` vs
   `torchvision.models.resnet18()` (if available). Which is larger?
   ResNet18 achieves comparable or better ImageNet accuracy with far fewer
   parameters than VGG16 — research this online and summarize in 2-3
   sentences why (hint: it's related to both architectural efficiency and
   depth-vs-width tradeoffs).
