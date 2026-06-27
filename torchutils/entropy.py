import torch

from .common_types import *


__all__ = ["shannons"]


# Checking

def _is_uint8_tensor(x: Tensor) -> bool:
    return x.dtype == torch.uint8


# Functional

def shannons(x: Tensor) -> Tensor:
    if not _is_uint8_tensor(x):
        raise TypeError("shannons only supports uint8 tensors")
    counts = torch.bincount(x.ravel())
    p = counts[0 < counts].to(torch.float32) / x.numel()
    return -1 * torch.sum(torch.log2(p) * p)
