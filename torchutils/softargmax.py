import torch

from .common_types import *


__all__ = ["softargmax1d", "softargmax2d", "softargmax3d"]


# Helper Functions

def _softargmax(x: Tensor, beta: float) -> Tensor:
    """Differentiable soft-argmax over the spatial dimensions of ``x``.

    Args:
        x (Tensor): Heatmaps of shape ``[b, c, *spatial]``.
        beta (float): Inverse temperature applied before the softmax. Larger
            values sharpen the distribution toward a hard argmax.

    Returns:
        Tensor: Expected coordinates of shape ``[b, c, n]`` where ``n`` is the
        number of spatial dimensions, ordered to match tensor indexing
        (e.g. ``(y, x)`` for 2D, ``(d, h, w)`` for 3D).
    """
    spatial = x.shape[2:]

    # Joint softmax over every spatial location.
    p = x.flatten(2).mul(beta).softmax(-1)  # [b, c, S]

    # Coordinate grid on the same device/dtype as the activations.
    grids = torch.meshgrid(
        *[torch.arange(s, dtype=p.dtype, device=p.device) for s in spatial],
        indexing="ij",
    )
    grid = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # [S, n]

    return p @ grid  # [b, c, n]


# Functional

def softargmax1d(x: Tensor, beta: float = 1e+3) -> Tensor:
    """Differentiable soft-argmax over 1D heatmaps.

    Args:
        x (Tensor): Heatmaps of shape ``[b, c, w]``.
        beta (float, optional): Inverse temperature. Larger values approach a
            hard argmax; smaller values give a smoother, more averaged estimate.
            Default: ``1e3``.

    Returns:
        Tensor: Expected positions of shape ``[b, c, 1]``, as float indices.

    Raises:
        RuntimeError: If ``x`` is not 3D.

    Example:
        >>> x = torch.zeros(1, 1, 5)
        >>> x[0, 0, 3] = 1.0
        >>> softargmax1d(x)
        tensor([[[3.]]])
    """
    if x.ndim != 3:
        raise RuntimeError(f"softargmax1d expects a 3D input [b, c, w], got {x.ndim}D")
    return _softargmax(x, beta)


def softargmax2d(x: Tensor, beta: float = 1e+3) -> Tensor:
    """Differentiable soft-argmax over 2D heatmaps, e.g. keypoint heatmaps.

    Args:
        x (Tensor): Heatmaps of shape ``[b, c, h, w]``.
        beta (float, optional): Inverse temperature. Larger values approach a
            hard argmax; smaller values give a smoother, more averaged estimate.
            Default: ``1e3``.

    Returns:
        Tensor: Expected coordinates of shape ``[b, c, 2]`` in ``(y, x)``
        order, as float pixel indices.

    Raises:
        RuntimeError: If ``x`` is not 4D.

    Example:
        >>> x = torch.zeros(1, 1, 4, 4)
        >>> x[0, 0, 1, 2] = 1.0
        >>> softargmax2d(x)
        tensor([[[1., 2.]]])
    """
    if x.ndim != 4:
        raise RuntimeError(f"softargmax2d expects a 4D input [b, c, h, w], got {x.ndim}D")
    return _softargmax(x, beta)


def softargmax3d(x: Tensor, beta: float = 1e+3) -> Tensor:
    """Differentiable soft-argmax over 3D heatmaps, e.g. volumetric keypoints.

    Args:
        x (Tensor): Heatmaps of shape ``[b, c, d, h, w]``.
        beta (float, optional): Inverse temperature. Larger values approach a
            hard argmax; smaller values give a smoother, more averaged estimate.
            Default: ``1e3``.

    Returns:
        Tensor: Expected coordinates of shape ``[b, c, 3]`` in ``(d, h, w)``
        order, as float indices.

    Raises:
        RuntimeError: If ``x`` is not 5D.

    Example:
        >>> x = torch.zeros(1, 1, 3, 4, 5)
        >>> x[0, 0, 2, 1, 4] = 1.0
        >>> softargmax3d(x)
        tensor([[[2., 1., 4.]]])
    """
    if x.ndim != 5:
        raise RuntimeError(f"softargmax3d expects a 5D input [b, c, d, h, w], got {x.ndim}D")
    return _softargmax(x, beta)
