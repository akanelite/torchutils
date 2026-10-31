import torch
import torch.nn as nn

from .common_types import Tensor


__all__ = ["poly_inr_coord_sampler", "poly_inr_s", "poly_inr_l"]


# Functional

def poly_inr_coord_sampler(size: tuple[int, int], device=None, dtype=None) -> Tensor:
    factory_kwargs = dict(device=device, dtype=dtype)
    tensors = [
        torch.linspace(0, 1, size[0], **factory_kwargs),
        torch.linspace(0, 1, size[1], **factory_kwargs),
    ]
    tensors = torch.stack(torch.meshgrid(*tensors, indexing="ij"), -1).reshape(-1, 2)
    tensors = torch.cat([tensors, torch.ones(tensors.size(0), 1, **factory_kwargs).reshape(-1, 1)], -1)
    return tensors


# Modules

class PINRBlock(nn.Module):

    def __init__(self, dim: int, affine: Tensor, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.affine = affine
        self.linear = nn.Linear(dim, dim, **factory_kwargs)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, r: Tensor | None = None) -> Tensor:
        x = x @ self.affine.T
        if r is not None:
            x = x * r
        x = self.linear(x)
        x = self.activation(x)
        return x


class PINR(nn.Module):

    def __init__(self, dim: int, levels: int = 10, device=None, dtype=None) -> None:
        factory_kwargs = dict(device=device, dtype=dtype)
        super().__init__()
        self.affine = torch.zeros(dim, 3, **factory_kwargs).normal_()
        self.layers = nn.ModuleList()
        for _ in range(levels):
            self.layers.append(PINRBlock(dim, self.affine, **factory_kwargs))
        self.head = nn.Sequential(nn.Linear(dim, 3, **factory_kwargs), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        r = self.layers[0](x)
        for layer in self.layers[1:]:
            r = layer(x, r)
        x = self.head(r)
        return x


# Interfaces

def poly_inr_s(dim: int = 512, **kwargs) -> PINR:
    return PINR(dim, **kwargs)


def poly_inr_l(dim: int = 1024, **kwargs) -> PINR:
    return PINR(dim, **kwargs)
