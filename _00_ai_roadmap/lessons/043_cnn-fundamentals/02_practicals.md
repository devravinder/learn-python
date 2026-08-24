# 02 — Practicals: CNN Fundamentals

1. Implement 2D convolution completely from scratch in NumPy (no
   `nn.Conv2d`): given a 2D input array and a small kernel (e.g. 3x3),
   slide the kernel and compute the output via nested loops (no need to
   vectorize fully — clarity over speed here). Test it on a simple 6x6
   input with a hand-designed vertical-edge-detector kernel
   (`[[1,0,-1],[1,0,-1],[1,0,-1]]`) applied to an image with a clear
   vertical edge (e.g. left half all 0s, right half all 1s) — confirm the
   output's magnitude peaks right at the edge column and is 0 in the flat
   regions on either side (the sign depends on which side is brighter and
   the kernel's own sign convention — check yours rather than assuming
   positive).

2. Compare your from-scratch convolution's output to `torch.nn.functional.conv2d`
   on the same input/kernel (reshape appropriately: `(1,1,H,W)` for input,
   `(1,1,kh,kw)` for kernel). Confirm they match.

3. Compute output shapes by hand for: a `32x32` input through
   `Conv2d(kernel_size=3, padding=1, stride=1)`, then `MaxPool2d(2)`, then
   `Conv2d(kernel_size=3, padding=1, stride=1)`, then `MaxPool2d(2)`. Verify
   your hand calculation against PyTorch by actually running a dummy tensor
   through the layers and checking `.shape`.

4. Visualize what a specific hand-designed kernel "sees": apply a horizontal-
   edge kernel, a vertical-edge kernel, and a blur kernel (`[[1,1,1],[1,1,1],[1,1,1]]/9`)
   to a real image (any grayscale image you have, or one from
   `sklearn.datasets.load_digits`) and plot all 3 outputs side by side with
   the original.

5. Train a small CNN (`Conv2d(1,8,3,padding=1), ReLU, MaxPool2d(2), Conv2d(8,16,3,padding=1), ReLU, Flatten, Linear`)
   on the digits dataset (like Project 006) and visualize the learned first-
   layer filter weights (`model[0].weight.detach()`, shape `(8,1,3,3)`) as
   8 small 3x3 images. Do any resemble recognizable edge/pattern detectors?

6. Compute the receptive field size (in original-image pixels) of a unit
   after 2 stacked `Conv2d(kernel_size=3, padding=1, stride=1)` layers with
   no pooling between them (hint: each additional 3x3 conv layer adds 2 to
   the receptive field's width/height). Confirm your answer by reasoning
   about which input pixels can influence a single output unit 2 layers
   deep.
