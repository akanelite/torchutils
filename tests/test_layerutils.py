from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from torchutils import IntermediateLayerGetter  # must be exported from the package


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(3, 4, 3, padding=1)),
            ("relu", nn.ReLU()),
        ]))
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        x = self.block(x)
        x = x.mean((2, 3))
        return self.head(x)


def test_extracts_nested_intermediate_features():
    model = Net()
    getter = IntermediateLayerGetter(model, {"block.conv": "feat", "block.relu": "act"})
    feats = getter(torch.randn(1, 3, 8, 8))
    assert set(feats.keys()) == {"feat", "act"}
    assert feats["feat"].shape == (1, 4, 8, 8)


def test_missing_layer_raises_clear_error():
    getter = IntermediateLayerGetter(Net(), {"block.nope": "x"})
    with pytest.raises(AttributeError, match="not found"):
        getter(torch.randn(1, 3, 8, 8))


def test_repeated_label_accumulates_into_list():
    getter = IntermediateLayerGetter(Net(), {"block.conv": "x", "block.relu": "x"})
    feats = getter(torch.randn(1, 3, 8, 8))
    assert isinstance(feats["x"], list)
    assert len(feats["x"]) == 2


def test_hooks_removed_after_forward():
    model = Net()
    getter = IntermediateLayerGetter(model, {"block.conv": "feat"})
    getter(torch.randn(1, 3, 8, 8))
    assert len(model.block.conv._forward_hooks) == 0


def test_hooks_removed_even_when_forward_raises():
    model = Net()
    getter = IntermediateLayerGetter(model, {"block.conv": "feat"})
    with pytest.raises(RuntimeError):
        getter(torch.randn(1, 5, 8, 8))  # wrong channel count -> conv raises
    assert len(model.block.conv._forward_hooks) == 0
