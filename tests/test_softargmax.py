import pytest
import torch

from torchutils.softargmax import softargmax1d, softargmax2d, softargmax3d


def _peak(shape, idx):
    """A heatmap that is sharply peaked at ``idx`` (and flat elsewhere)."""
    t = torch.full(shape, -1e4)
    t[idx] = 1e4
    return t


# Peak recovery: a sharp peak should resolve to its index.

def test_softargmax1d_recovers_peak():
    x = torch.stack([_peak((1, 6), (0, 2)), _peak((1, 6), (0, 5))], dim=1)  # [1, 2, 6]
    out = softargmax1d(x)
    assert out.shape == (1, 2, 1)
    assert torch.allclose(out.flatten(), torch.tensor([2.0, 5.0]), atol=1e-3)


def test_softargmax2d_recovers_peak_in_yx_order():
    hm = _peak((1, 1, 5, 7), (0, 0, 3, 1))  # row (y) = 3, col (x) = 1
    out = softargmax2d(hm)
    assert out.shape == (1, 1, 2)
    assert torch.allclose(out.flatten(), torch.tensor([3.0, 1.0]), atol=1e-3)


def test_softargmax3d_recovers_peak_in_dhw_order():
    hm = _peak((1, 1, 4, 5, 6), (0, 0, 2, 1, 4))
    out = softargmax3d(hm)
    assert out.shape == (1, 1, 3)
    assert torch.allclose(out.flatten(), torch.tensor([2.0, 1.0, 4.0]), atol=1e-3)


# Output shape: [b, c, n] for any number of channels.

@pytest.mark.parametrize("fn, shape, n", [
    (softargmax1d, (4, 17, 9), 1),
    (softargmax2d, (4, 17, 8, 8), 2),
    (softargmax3d, (4, 17, 4, 5, 6), 3),
])
def test_output_shape_multichannel(fn, shape, n):
    out = fn(torch.randn(*shape))
    assert out.shape == (shape[0], shape[1], n)


# dtype/device follow the input (regression: weight used to be CPU/float32).

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_dtype_is_preserved(dtype):
    out = softargmax2d(torch.randn(1, 3, 8, 8, dtype=dtype))
    assert out.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_runs_on_cuda():
    out = softargmax2d(torch.randn(1, 2, 8, 8, device="cuda"))
    assert out.device.type == "cuda"


# Robustness.

def test_non_contiguous_input():
    base = torch.randn(2, 3, 8, 8, 2)
    nc = base[..., 0]  # non-contiguous [2, 3, 8, 8]
    assert not nc.is_contiguous()
    out = softargmax2d(nc)
    assert out.shape == (2, 3, 2)


def test_gradient_flows():
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    softargmax2d(x).sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


# Input validation.

@pytest.mark.parametrize("fn, bad_shape", [
    (softargmax1d, (1, 1, 4, 4)),
    (softargmax2d, (1, 1, 4)),
    (softargmax3d, (1, 1, 4, 4)),
])
def test_wrong_ndim_raises(fn, bad_shape):
    with pytest.raises(RuntimeError):
        fn(torch.randn(*bad_shape))
