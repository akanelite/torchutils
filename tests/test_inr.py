import pytest
import torch

import torchutils
from torchutils.inr import poly_inr_coord_sampler, poly_inr_l, poly_inr_s


# Coordinate sampler: [h * w, 3] grid of (y, x, 1) in [0, 1].

def test_coord_sampler_shape_and_range():
    coords = poly_inr_coord_sampler((4, 6))
    assert coords.shape == (24, 3)
    assert torch.all(coords[:, :2] >= 0) and torch.all(coords[:, :2] <= 1)
    assert torch.all(coords[:, 2] == 1)


def test_coord_sampler_ij_order():
    coords = poly_inr_coord_sampler((3, 5)).view(3, 5, 3)
    assert torch.allclose(coords[0, 0, :2], torch.tensor([0.0, 0.0]))
    assert torch.allclose(coords[-1, 0, :2], torch.tensor([1.0, 0.0]))
    assert torch.allclose(coords[0, -1, :2], torch.tensor([0.0, 1.0]))


def test_coord_sampler_dtype():
    assert poly_inr_coord_sampler((2, 2), dtype=torch.float64).dtype == torch.float64


# Models: map [n, 3] coordinates to [n, 3] RGB in (0, 1).

@pytest.mark.parametrize("factory, dim", [(poly_inr_s, 512), (poly_inr_l, 1024)])
def test_model_default_dims(factory, dim):
    model = factory(levels=2)
    assert model.affine.shape == (dim, 3)
    assert len(model.layers) == 2


def test_model_forward_shape_and_range():
    model = poly_inr_s(dim=32, levels=3)
    out = model(poly_inr_coord_sampler((8, 8)))
    assert out.shape == (64, 3)
    assert torch.all(out > 0) and torch.all(out < 1)


def test_model_backward():
    model = poly_inr_s(dim=16, levels=2)
    model(poly_inr_coord_sampler((4, 4))).sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_exported_from_package():
    for name in ["poly_inr_coord_sampler", "poly_inr_s", "poly_inr_l"]:
        assert hasattr(torchutils, name)


# The shared affine projection is a buffer: it moves with the module and is saved.

def test_affine_in_state_dict_and_restored():
    src = poly_inr_s(dim=16, levels=2)
    dst = poly_inr_s(dim=16, levels=2)
    assert "affine" in src.state_dict()
    dst.load_state_dict(src.state_dict())
    x = poly_inr_coord_sampler((4, 4))
    assert torch.equal(src(x), dst(x))


def test_affine_follows_to():
    model = poly_inr_s(dim=16, levels=2).to(torch.float64)
    assert model.affine.dtype == torch.float64
    out = model(poly_inr_coord_sampler((4, 4), dtype=torch.float64))
    assert out.dtype == torch.float64
