"""Stable neuron IDs for the original AWN DynamicMLP width transitions.

Existing prefix neurons retain their identity when a layer shrinks or grows.
A newly allocated neuron receives a new ID even if it reuses an old slot.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class NeuronIdentityTracker:
    namespace: str
    width: int
    counter: int = 0
    ids: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.width < 1 or not self.namespace:
            raise ValueError("Initial neuron width and namespace must be valid")
        if self.ids:
            raise ValueError("Identity tracker creates its own initial IDs")
        self._grow(self.width)

    def _grow(self, count: int) -> None:
        for _ in range(count):
            self.ids.append(f"{self.namespace}:neuron:{self.counter:06d}")
            self.counter += 1

    def resize(self, width: int) -> None:
        if width < 1:
            raise ValueError("No valid active neuron prefix")
        old_width = len(self.ids)
        if width == old_width:
            return
        if width < old_width:
            removed = self.ids[width:]
            self.ids = self.ids[:width]
            self.events.append({"event": "retire", "ids": removed,
                                "old_width": old_width, "new_width": width})
        else:
            self._grow(width - old_width)
            self.events.append({"event": "activate_new", "ids": self.ids[old_width:],
                                "old_width": old_width, "new_width": width})

    def snapshot(self) -> tuple[str, ...]:
        return tuple(self.ids)


@contextmanager
def track_awn_width(model, namespace: str) -> Iterator[NeuronIdentityTracker]:
    """Monitor the real first-hidden-layer change_shape calls.

    Only a one-hidden-layer DynamicMLP is supported. The original method is
    restored upon exit. Live BT permutations additionally require identity
    updates at the permutation boundary; this tracker does not infer them.
    """
    dyn = model.dyn_model
    if getattr(dyn, "num_layers", None) != 1:
        raise ValueError("Only one-hidden-layer DynamicMLP is supported")
    initial_width = int(dyn.get_layer(0).weight.shape[0])
    tracker = NeuronIdentityTracker(namespace=namespace, width=initial_width)
    original = dyn.change_shape
    had_override = "change_shape" in vars(dyn)

    def observed_change_shape(layer_id: int, num_neurons: int, change_output: bool):
        before = int(dyn.get_layer(0).weight.shape[0])
        result = original(layer_id=layer_id, num_neurons=num_neurons,
                          change_output=change_output)
        after = int(dyn.get_layer(0).weight.shape[0])
        if layer_id == 0 and change_output and after != before:
            tracker.resize(after)
        return result

    dyn.change_shape = observed_change_shape
    try:
        yield tracker
    finally:
        if had_override:
            dyn.change_shape = original
        else:
            delattr(dyn, "change_shape")
