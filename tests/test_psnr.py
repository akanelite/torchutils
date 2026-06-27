import math

import pytest
import torch

from torchutils.psnr import psnr


def test_identical_inputs_return_inf():
    a = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    assert torch.isinf(psnr(a, a))


def test_known_value_with_unit_mse():
    a = torch.zeros(4, 4)
    b = torch.ones(4, 4)  # mse == 1 -> 20*log10(max_v) - 0
    out = psnr(a, b, max_v=255)
    assert torch.isclose(out, torch.tensor(20 * math.log10(255)), atol=1e-3)


def test_higher_for_smaller_error():
    a = torch.zeros(8, 8)
    near = torch.full((8, 8), 1.0)
    far = torch.full((8, 8), 10.0)
    assert psnr(a, near) > psnr(a, far)


def test_shape_mismatch_raises():
    with pytest.raises(RuntimeError):
        psnr(torch.zeros(3, 3), torch.zeros(3, 4))


def test_accepts_integer_inputs():
    a = torch.zeros(4, 4, dtype=torch.uint8)
    b = torch.ones(4, 4, dtype=torch.uint8)
    out = psnr(a, b, max_v=255)
    assert torch.isclose(out, torch.tensor(20 * math.log10(255)), atol=1e-3)
