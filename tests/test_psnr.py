import math

import pytest
import torch

from torchutils.psnr import psnr


# Shape handling: single image -> scalar, batch -> per-sample.

def test_single_image_returns_scalar():
    out = psnr(torch.rand(3, 8, 8), torch.rand(3, 8, 8))
    assert out.ndim == 0


def test_identical_single_image_is_inf():
    a = torch.rand(3, 8, 8)
    assert torch.isinf(psnr(a, a))


def test_batch_returns_one_value_per_sample():
    out = psnr(torch.rand(4, 3, 8, 8), torch.rand(4, 3, 8, 8))
    assert out.shape == (4,)


def test_batch_values_are_independent_per_sample():
    a = torch.rand(2, 3, 8, 8)
    b = a.clone()
    b[1] = b[1] + 0.5  # only the second sample differs
    out = psnr(a, b)
    assert torch.isinf(out[0])
    assert torch.isfinite(out[1])


# max_v inference from dtype.

def test_float_default_max_v_is_one():
    # mse == 1, max_v == 1 -> 10*log10(1) == 0 dB
    out = psnr(torch.zeros(3, 4, 4), torch.ones(3, 4, 4))
    assert torch.isclose(out, torch.tensor(0.0), atol=1e-4)


def test_uint8_default_max_v_is_255():
    a = torch.zeros(3, 4, 4, dtype=torch.uint8)
    b = torch.ones(3, 4, 4, dtype=torch.uint8)  # mse == 1
    assert torch.isclose(psnr(a, b), torch.tensor(20 * math.log10(255)), atol=1e-3)


def test_uint16_default_max_v_is_65535():
    a = torch.zeros(3, 4, 4, dtype=torch.uint16)
    b = torch.ones(3, 4, 4, dtype=torch.uint16)  # mse == 1
    assert torch.isclose(psnr(a, b), torch.tensor(20 * math.log10(65535)), atol=1e-3)


def test_explicit_max_v_overrides_inference():
    out = psnr(torch.zeros(3, 4, 4), torch.ones(3, 4, 4), max_v=255)
    assert torch.isclose(out, torch.tensor(20 * math.log10(255)), atol=1e-3)


def test_higher_for_smaller_error():
    a = torch.zeros(3, 8, 8)
    near = torch.full((3, 8, 8), 0.1)
    far = torch.full((3, 8, 8), 0.9)
    assert psnr(a, near) > psnr(a, far)


# Validation.

def test_shape_mismatch_raises():
    with pytest.raises(RuntimeError):
        psnr(torch.zeros(3, 3, 3), torch.zeros(3, 3, 4))


def test_wrong_ndim_raises():
    with pytest.raises(RuntimeError):
        psnr(torch.zeros(4, 4), torch.zeros(4, 4))  # 2D is not supported
