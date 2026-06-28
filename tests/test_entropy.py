import pytest
import torch

from torchutils.entropy import shannons


def test_uniform_256_symbols_is_8_bits():
    x = torch.arange(256, dtype=torch.uint8)  # every symbol once
    assert torch.isclose(shannons(x), torch.tensor(8.0), atol=1e-5)


def test_constant_input_is_zero_and_nonnegative():
    out = shannons(torch.zeros(100, dtype=torch.uint8))
    assert out.item() == 0.0          # not -0.0
    assert out.item() >= 0.0


def test_two_symbols_half_half_is_1_bit():
    x = torch.tensor([0, 1] * 50, dtype=torch.uint8)
    assert torch.isclose(shannons(x), torch.tensor(1.0), atol=1e-5)


def test_handles_multidimensional_input():
    img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    out = shannons(img)
    assert out.ndim == 0
    assert 0.0 <= out.item() <= 8.0


def test_non_uint8_raises():
    with pytest.raises(TypeError):
        shannons(torch.arange(10, dtype=torch.int32))


def test_empty_raises():
    with pytest.raises(ValueError):
        shannons(torch.empty(0, dtype=torch.uint8))
