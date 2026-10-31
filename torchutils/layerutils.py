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
    """Wrap a model and return the outputs of selected inner layers.

    Calling the wrapper runs the full forward pass of ``model`` and collects the
    outputs of the requested submodules through temporary forward hooks. The
    model's own return value is discarded. Hooks are removed after every call,
    so the wrapped model is left unchanged.

    Args:
        model (nn.Module): The model to inspect.
        return_layers (dict[str, str]): Maps a submodule path to the key under
            which its output is returned. Paths use attribute names joined by
            dots, as in ``model.named_modules()`` (e.g. ``"layer1.0.conv"``).

    Example:
        >>> model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        >>> getter = IntermediateLayerGetter(model, {"0": "hidden", "2": "logits"})
        >>> features = getter(torch.randn(2, 8))
        >>> {k: tuple(v.shape) for k, v in features.items()}
        {'hidden': (2, 16), 'logits': (2, 4)}
    """

    def __init__(self, model: nn.Module, return_layers: dict[str, str]) -> None:
        super().__init__()
        self._model = model
        self.return_layers = return_layers

    def forward(self, *args: Any, **kwargs: Any) -> OrderedDict[str, torch.Tensor]:
        """Run the wrapped model and collect the requested layer outputs.

        Args:
            *args: Positional arguments passed to the wrapped model.
            **kwargs: Keyword arguments passed to the wrapped model.

        Returns:
            OrderedDict[str, Tensor]: Layer outputs keyed by label, in the order
            the layers ran. A layer called more than once in one forward pass
            yields a list with one output per call.

        Raises:
            AttributeError: If a path in ``return_layers`` does not exist in
                the wrapped model.
        """
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
