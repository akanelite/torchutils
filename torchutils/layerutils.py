from collections import OrderedDict
from typing import Any

import functools

import torch
import torch.nn as nn


__all__ = ["IntermediateLayerGetter"]


# Helper Functions

def reduced_getattr(obj: Any, attr: str) -> Any:
    """Resolve a dotted attribute path (e.g. ``"block.conv"``) on ``obj``.

    Raises ``AttributeError`` if any segment is missing.
    """
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


# Modules

class IntermediateLayerGetter(nn.Module):

    def __init__(self, model: nn.Module, return_layers: dict[str, str]) -> None:
        super().__init__()
        self._model = model
        self.return_layers = return_layers

    def forward(self, *args: Any, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        features = OrderedDict()
        handlers = []

        def hook(module, inputs, output, label):
            if label in features:
                if isinstance(features[label], list):
                    features[label].append(output)
                else:
                    features[label] = [features[label], output]
            else:
                features[label] = output

        for name, label in self.return_layers.items():
            try:
                layer = reduced_getattr(self._model, name)
            except AttributeError:
                raise AttributeError(f"return layer {name!r} not found in model") from None

            handlers.append(layer.register_forward_hook(functools.partial(hook, label=label)))

        try:
            self._model(*args, **kwargs)
        finally:
            for handler in handlers:
                handler.remove()

        return features
