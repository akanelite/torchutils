import torch
import torch.nn as nn

from .common_types import Tensor


__all__ = ["poly_inr_coord_sampler", "poly_inr_s", "poly_inr_l"]


# Functional

def poly_inr_coord_sampler(size: tuple[int, int], device=None, dtype=None) -> Tensor:
    """Build the input coordinates for a Poly-INR model.

    Samples a regular grid over ``[0, 1] x [0, 1]`` in row-major order and
    appends a constant ``1`` to each point, giving homogeneous coordinates
    ``(y, x, 1)`` that :class:`PINR` projects with its affine matrix.

    Args:
        size (tuple[int, int]): Grid size as ``(height, width)``.
        device (torch.device, optional): Device of the returned tensor.
        dtype (torch.dtype, optional): Floating point dtype of the returned tensor.

    Returns:
        Tensor: Coordinates of shape ``[height * width, 3]``.

    Example:
        >>> poly_inr_coord_sampler((2, 3))
        tensor([[0.0000, 0.0000, 1.0000],
                [0.0000, 0.5000, 1.0000],
                [0.0000, 1.0000, 1.0000],
                [1.0000, 0.0000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 1.0000, 1.0000]])
    """
    factory_kwargs = dict(device=device, dtype=dtype)
    tensors = [
        torch.linspace(0, 1, size[0], **factory_kwargs),
        torch.linspace(0, 1, size[1], **factory_kwargs),
    ]
    tensors = torch.stack(torch.meshgrid(*tensors, indexing="ij"), -1).reshape(-1, 2)
    tensors = torch.cat([tensors, torch.ones(tensors.size(0), 1, **factory_kwargs).reshape(-1, 1)], -1)
    return tensors


# Modules

class PINRBlock(nn.Module):
    """One level of a Poly-INR network.

    Projects the coordinates with the shared affine matrix, multiplies the
    result element-wise with the previous level's features (raising the
    polynomial order by one), then applies a linear layer and LeakyReLU.

    Args:
        dim (int): Feature dimension.
        device (torch.device, optional): Device of the parameters.
        dtype (torch.dtype, optional): Dtype of the parameters.
    """

    def __init__(self, dim: int, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.linear = nn.Linear(dim, dim, **factory_kwargs)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, affine: Tensor, r: Tensor | None = None) -> Tensor:
        x = x @ affine.T
        if r is not None:
            x = x * r
        x = self.linear(x)
        x = self.activation(x)
        return x


class PINR(nn.Module):
    """Poly-INR: an implicit neural representation that maps 2D coordinates to RGB.

    Instead of sinusoidal encodings, each level multiplies an affine projection
    of the input coordinates into the running features, so the network is a
    polynomial of the coordinates whose order grows with ``levels``. Fit one
    model per image by regressing its pixel colors.

    Args:
        dim (int): Feature dimension of every level.
        levels (int, optional): Number of :class:`PINRBlock` levels. Default: ``10``.
        device (torch.device, optional): Device of the parameters.
        dtype (torch.dtype, optional): Dtype of the parameters.

    Shape:
        - Input: ``[N, 3]`` homogeneous coordinates from
          :func:`poly_inr_coord_sampler`.
        - Output: ``[N, 3]`` RGB values in ``[0, 1]``.

    Example:
        >>> model = PINR(dim=64, levels=4)
        >>> coords = poly_inr_coord_sampler((16, 16))
        >>> rgb = model(coords)
        >>> rgb.shape
        torch.Size([256, 3])
        >>> image = rgb.T.reshape(3, 16, 16)  # back to [C, H, W]
    """

    def __init__(self, dim: int, levels: int = 10, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        # Fixed random projection shared by every block. Registered as a buffer so it
        # follows .to() / .cuda() and is saved in the state_dict.
        self.register_buffer("affine", torch.zeros(dim, 3, **factory_kwargs).normal_())
        self.layers = nn.ModuleList()
        for _ in range(levels):
            self.layers.append(PINRBlock(dim, **factory_kwargs))
        self.head = nn.Sequential(nn.Linear(dim, 3, **factory_kwargs), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        r = self.layers[0](x, self.affine)
        for layer in self.layers[1:]:
            r = layer(x, self.affine, r)
        x = self.head(r)
        return x


# Interfaces

def poly_inr_s(dim: int = 512, **kwargs) -> PINR:
    """Create the small Poly-INR configuration (``dim=512``).

    Args:
        dim (int, optional): Feature dimension. Default: ``512``.
        **kwargs: Passed to :class:`PINR` (``levels``, ``device``, ``dtype``).

    Returns:
        PINR: The model.

    Example:
        >>> model = poly_inr_s()
        >>> model(poly_inr_coord_sampler((8, 8))).shape
        torch.Size([64, 3])
    """
    return PINR(dim, **kwargs)


def poly_inr_l(dim: int = 1024, **kwargs) -> PINR:
    """Create the large Poly-INR configuration (``dim=1024``).

    Args:
        dim (int, optional): Feature dimension. Default: ``1024``.
        **kwargs: Passed to :class:`PINR` (``levels``, ``device``, ``dtype``).

    Returns:
        PINR: The model.
    """
    return PINR(dim, **kwargs)
