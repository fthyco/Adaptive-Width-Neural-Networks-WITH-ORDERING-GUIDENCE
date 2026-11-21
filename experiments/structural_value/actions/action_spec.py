"""Explicit actions; resize and joint are reserved for later research gates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json


class ActionKind(str, Enum):
    SKIP = "skip"
    RANDOM_POS = "random_positional"
    WEIGHT_POS = "weight_positional"
    GRADIENT_POS = "gradient_positional"
    RESIZE = "resize"
    JOINT = "joint"


@dataclass(frozen=True)
class ActionSpec:
    kind: ActionKind
    gate_mode: str = "positional"
    action_seed: int | None = None
    target_widths: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.gate_mode not in ("positional", "neuron_coupled"):
            raise ValueError("Unknown gate semantics")
        if self.kind is ActionKind.SKIP:
            if self.action_seed is not None or self.target_widths is not None or self.gate_mode != "positional":
                raise ValueError("Skip cannot carry an intervention payload")
        elif self.kind is ActionKind.RANDOM_POS and self.action_seed is None:
            raise ValueError("Random ordering requires an explicit action seed")
        if self.kind in (ActionKind.RESIZE, ActionKind.JOINT):
            if self.target_widths is None or not self.target_widths or min(self.target_widths) < 1:
                raise ValueError("Resize requires positive integer target widths")
        elif self.target_widths is not None:
            raise ValueError("Reorder actions cannot specify a target width")

    @property
    def payload(self) -> dict:
        return {
            "kind": self.kind.value,
            "gate_mode": self.gate_mode,
            "action_seed": self.action_seed,
            "target_widths": list(self.target_widths) if self.target_widths else None,
        }

    @property
    def action_id(self) -> str:
        return sha256(json.dumps(self.payload, sort_keys=True).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class ActionMeasurement:
    no_op: bool
    evidence_seconds: float = 0.0
    intervention_seconds: float = 0.0
    details: dict | None = None

    def __post_init__(self) -> None:
        if min(self.evidence_seconds, self.intervention_seconds) < 0:
            raise ValueError("Negative measured action time")
