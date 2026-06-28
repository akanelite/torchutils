import torch

from .common_types import *


__all__ = ["psnr"]


# Functional

def psnr(x: Tensor, other: Tensor, max_v: float | None = None) -> Tensor:
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Higher values indicate better quality (lower distortion). Identical inputs
    yield ``inf`` (computed per sample for batched inputs).

    Args:
        x (Tensor): The input image(s), shaped ``[C, H, W]`` or ``[B, C, H, W]``.
        other (Tensor): The reference image(s), same shape as ``x``.
        max_v (float, optional): The maximum possible signal value. When omitted
            it is inferred from ``x``'s dtype: ``1.0`` for floating point
            (assumed normalized to ``[0, 1]``) and the dtype maximum for
            integers (e.g. ``255`` for ``uint8``, ``65535`` for ``uint16``).

    Returns:
        Tensor: PSNR in decibels (dB). A scalar for a single ``[C, H, W]`` image,
        or a ``[B]`` tensor of per-sample values for a ``[B, C, H, W]`` batch.
    """
    if x.shape != other.shape:
        raise RuntimeError(f"Shape of {x.shape} does not match {other.shape}")

    if x.ndim not in (3, 4):
        raise RuntimeError(f"psnr expects [C, H, W] or [B, C, H, W], got {x.ndim}D")

    if max_v is None:
        max_v = 1.0 if x.dtype.is_floating_point else torch.iinfo(x.dtype).max

    x = x.to(torch.float32)
    other = other.to(torch.float32)

    se = (x - other) ** 2
    mse = se.mean(dim=(1, 2, 3)) if x.ndim == 4 else se.mean()

    # 10 * log10(max_v ** 2 / mse); identical inputs (mse == 0) give +inf.
    return mse.reciprocal().mul(max_v ** 2).log10().mul(10)
