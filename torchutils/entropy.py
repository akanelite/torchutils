import torch

from .common_types import *


__all__ = ["shannons"]


# Checking

def _is_uint8_tensor(x: Tensor) -> bool:
    return x.dtype == torch.uint8


# Functional

def shannons(x: Tensor) -> Tensor:
    """Compute the Shannon entropy of a uint8 tensor, in bits per symbol.

    Every element is treated as one sample of a byte-valued symbol, and the
    entropy of their empirical distribution is returned. The result lies in
    ``[0, 8]``: ``0`` for a constant tensor and ``8`` when all 256 values occur
    equally often.

    Args:
        x (Tensor): A non-empty tensor of dtype torch.uint8, of any shape.

    Returns:
        Tensor: A float32 scalar holding the entropy in bits.

    Raises:
        TypeError: If ``x`` is not a uint8 tensor.
        ValueError: If ``x`` is empty.

    Example:
        >>> shannons(torch.tensor([0, 0, 1, 1], dtype=torch.uint8))
        tensor(1.)
        >>> shannons(torch.arange(256, dtype=torch.uint8))
        tensor(8.)
    """
    if not _is_uint8_tensor(x):
        raise TypeError("shannons only supports uint8 tensors")
    if x.numel() == 0:
        raise ValueError("shannons requires a non-empty tensor")
    counts = torch.bincount(x.ravel())
    p = counts[counts > 0].to(torch.float32) / x.numel()
    return torch.sum(-p * torch.log2(p)).clamp_min(0.0)
