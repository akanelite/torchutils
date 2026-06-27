import math

import torch

from .common_types import *


__all__ = ["psnr"]


# Checking

def _is_float32_tensor(x: Tensor) -> bool:
    return x.dtype == torch.float32


# Functional

def psnr(x: Tensor, other: Tensor, max_v: float = 255) -> Tensor:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two tensors.

    PSNR is a widely used metric for measuring the similarity between two images
    or signals. Higher values indicate better quality (lower distortion).

    Note:
        If the mean squared error (MSE) is zero, indicating identical inputs,
        the function returns `float('inf')`.

    Args:
        x (Tensor): The input tensor (e.g., predicted image).
        other (Tensor): The reference tensor (e.g., ground truth image).
        max_v (float, optional): The maximum possible signal value. Default is 255.

    Returns:
        Tensor: A scalar tensor containing the PSNR value in decibels (dB).
    """
    if x.shape != other.shape:
        raise RuntimeError(f"Shape of {x.shape} does not match {other.shape}")

    if not _is_float32_tensor(x):
        x = x.to(torch.float32)

    if not _is_float32_tensor(other):
        other = other.to(torch.float32)

    mse = torch.mean((x - other) ** 2)

    if mse == 0:
        return torch.tensor(float("inf"), device=x.device)

    return 20 * math.log10(max_v) - 10 * torch.log10(mse)
