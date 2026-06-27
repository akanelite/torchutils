import torch

from .common_types import *


__all__ = ["i8u8", "u8i8"]


# Checking

def _is_int8_tensor(x: Tensor) -> bool:
    return x.dtype == torch.int8


def _is_uint8_tensor(x: Tensor) -> bool:
    return x.dtype == torch.uint8


def _is_tracing_or_scripting() -> bool:
    return torch.jit.is_scripting() or torch.jit.is_tracing()


# Functional

def i8u8(x: Tensor) -> Tensor:
    """Convert int8 tensor to uint8 with +128 offset.

    This function maps signed int8 values [-128, 127] to the full range of
    unsigned uint8 [0, 255] by adding 128 to each element.

    When running in ONNX export or TorchScript tracing/scripting mode,
    uses explicit type casting to maintain compatibility.

    Args:
        x (Tensor): A tensor of dtype torch.int8.

    Returns:
        Tensor: A tensor of dtype torch.uint8, each element offset by +128.
    """
    if not _is_int8_tensor(x):
        raise TypeError("i8u8 only supports int8 tensors")
    if _is_tracing_or_scripting():
        return (x.to(torch.int16) + 128).to(torch.uint8)
    return x.view(torch.uint8) ^ 0b10000000


def u8i8(x: Tensor) -> Tensor:
    """Convert uint8 tensor to int8 with -128 offset.

    This function maps unsigned uint8 values [0, 255] back to the signed int8 range
    [-128, 127] by subtracting 128 from each element.

    When running in ONNX export or TorchScript tracing/scripting mode,
    uses explicit type casting to maintain compatibility.

    Args:
        x (Tensor): A tensor of dtype torch.uint8.

    Returns:
        Tensor: A tensor of dtype torch.int8, each element offset by -128.
    """
    if not _is_uint8_tensor(x):
        raise TypeError("u8i8 only supports uint8 tensors")
    if _is_tracing_or_scripting():
        return (x.to(torch.int16) - 128).to(torch.int8)
    return (x ^ 0b10000000).view(torch.int8)
