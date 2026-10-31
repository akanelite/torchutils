# Torchutils

A small collection of PyTorch utility functions and modules.
`torchutils` bundles a handful of lightweight helpers that come up repeatedly
when working with PyTorch tensors and models.

## Installation

Requirements
- Python 3.10+
- PyTorch 2.0+

Install directly from the repository:

```bash
pip install git+https://github.com/akanelite/torchutils.git
```

Or clone and install from source:

```bash
git clone https://github.com/akanelite/torchutils.git
cd torchutils
pip install .
```

Using [uv](https://github.com/astral-sh/uv):

```bash
uv add git+https://github.com/akanelite/torchutils.git
```

## Modules

| Module | Exports | Description |
| --- | --- | --- |
| `bitops` | `i8u8`, `u8i8` | Lossless int8 ↔ uint8 conversion with a ±128 offset |
| `entropy` | `shannons` | Shannon entropy of a uint8 tensor, in bits |
| `psnr` | `psnr` | Peak Signal-to-Noise Ratio for single images or batches |
| `softargmax` | `softargmax1d`, `softargmax2d`, `softargmax3d` | Differentiable argmax over heatmaps |
| `layerutils` | `IntermediateLayerGetter` | Collect outputs of inner layers from any model |
| `inr` | `poly_inr_s`, `poly_inr_l`, `poly_inr_coord_sampler` | Poly-INR implicit neural representation |

Everything is importable from the top-level package, e.g. `from torchutils import psnr`.
Each function and class has a docstring with arguments, shapes and examples;
use `help(torchutils.psnr)` to read it.

## Usage

### Bit operations

Shift int8 values into the uint8 range and back without loss, e.g. to store
quantized weights as bytes. Works under TorchScript and ONNX export.

```python
import torch
from torchutils import i8u8, u8i8

x = torch.tensor([-128, -1, 0, 127], dtype=torch.int8)
u = i8u8(x)          # tensor([  0, 127, 128, 255], dtype=torch.uint8)
assert torch.equal(u8i8(u), x)
```

### Entropy

```python
from torchutils import shannons

shannons(torch.arange(256, dtype=torch.uint8))  # tensor(8.), the maximum
```

### PSNR

`max_v` is inferred from the dtype: `1.0` for floats (assumed in `[0, 1]`),
`255` for `uint8`, and so on. Batched inputs return one value per sample.

```python
from torchutils import psnr

pred = torch.rand(4, 3, 64, 64)
target = torch.rand(4, 3, 64, 64)
psnr(pred, target)   # shape [4]
```

### Soft-argmax

Turn heatmaps into coordinates while staying differentiable, e.g. for
keypoint regression. Coordinates follow tensor indexing order (`(y, x)` in 2D).

```python
from torchutils import softargmax2d

heatmaps = torch.zeros(1, 1, 4, 4)
heatmaps[0, 0, 1, 2] = 1.0
softargmax2d(heatmaps)  # tensor([[[1., 2.]]])
```

Lower `beta` for a smoother estimate, raise it to get closer to a hard argmax.

### Intermediate layer outputs

Get features from inner layers without modifying the model. Layer paths are
the names shown by `model.named_modules()`.

```python
import torch.nn as nn
from torchutils import IntermediateLayerGetter

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
getter = IntermediateLayerGetter(model, {"0": "hidden", "2": "logits"})
features = getter(torch.randn(2, 8))
features["hidden"].shape  # torch.Size([2, 16])
```

### Poly-INR

Fit an image as a function from pixel coordinates to RGB.

```python
from torchutils import poly_inr_s, poly_inr_coord_sampler

image = torch.rand(3, 64, 64)                    # target, [C, H, W] in [0, 1]
coords = poly_inr_coord_sampler((64, 64))        # [H * W, 3]
target = image.reshape(3, -1).T                  # [H * W, 3]

model = poly_inr_s()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
for _ in range(1000):
    loss = (model(coords) - target).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

recon = model(coords).T.reshape(3, 64, 64)       # back to [C, H, W]
```

## Development

```bash
uv sync
uv run pytest
```

`pytest` also runs the examples in the docstrings (`--doctest-modules`), so
keep them correct when changing behavior.
