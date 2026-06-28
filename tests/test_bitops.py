import pytest
import torch

from torchutils.bitops import i8u8, u8i8


def test_roundtrip_covers_full_range():
    x = torch.arange(-128, 128, dtype=torch.int8)
    assert torch.equal(u8i8(i8u8(x)), x)


def test_i8u8_maps_to_full_uint8_range():
    x = torch.tensor([-128, -1, 0, 127], dtype=torch.int8)
    assert torch.equal(i8u8(x), torch.tensor([0, 127, 128, 255], dtype=torch.uint8))
    assert i8u8(x).dtype == torch.uint8


def test_u8i8_maps_back_to_int8_range():
    x = torch.tensor([0, 127, 128, 255], dtype=torch.uint8)
    assert torch.equal(u8i8(x), torch.tensor([-128, -1, 0, 127], dtype=torch.int8))
    assert u8i8(x).dtype == torch.int8


def test_i8u8_matches_arithmetic_reference():
    x = torch.arange(-128, 128, dtype=torch.int8)
    ref = (x.to(torch.int16) + 128).to(torch.uint8)
    assert torch.equal(i8u8(x), ref)


def test_i8u8_rejects_non_int8():
    with pytest.raises(TypeError):
        i8u8(torch.tensor([1], dtype=torch.uint8))


def test_u8i8_rejects_non_uint8():
    with pytest.raises(TypeError):
        u8i8(torch.tensor([1], dtype=torch.int8))
