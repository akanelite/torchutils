from collections import OrderedDict
from typing import Any

import functools

import torch
import torch.nn as nn


__all__ = ["IntermediateLayerGetter"]


# Helper Functions

def reduced_getattr(obj: Any, attr: str, *args: Any):
    def func(_obj, _attr):
        return getattr(_obj, _attr, *args)
    return functools.reduce(func, [obj] + attr.split("."))


# Modules

class IntermediateLayerGetter(nn.Module):

    def __init__(self, model: nn.Module, return_layers: dict[str, str]) -> None:
        super().__init__()
        self._model = model
        self.return_layers = return_layers

    def forward(self, *args: Any, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        features = OrderedDict()
        handlers = []

        for name, label in self.return_layers.items():
            layer = reduced_getattr(self._model, name)

            # noinspection PyUnusedLocal
            def hook(m, i, o, n):
                if n in features:
                    if isinstance(features[n], list):
                        features[n].append(o)
                    else:
                        features[n] = [features[n], o]
                else:
                    features[n] = o

            try:
                handler = layer.register_forward_hook(functools.partial(hook, n=label))
            except AttributeError:
                raise AttributeError(f"{name} not found")

            handlers.append(handler)

        self._model(*args, **kwargs)

        for handler in handlers:
            handler.remove()

        return features
