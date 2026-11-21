"""One-boundary time accounting; do not include offline teacher training."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class OnlineCostLedger:
    optimizer_steps: int = 0
    examples: int = 0
    evidence_seconds: float = 0.0
    intervention_seconds: float = 0.0
    policy_seconds: float = 0.0
    training_seconds: float = 0.0
    evaluation_seconds: float = 0.0

    @property
    def online_seconds(self) -> float:
        return (self.evidence_seconds + self.intervention_seconds +
                self.policy_seconds + self.training_seconds + self.evaluation_seconds)

    def step(self, batch_size: int, elapsed: float) -> None:
        if batch_size < 1 or elapsed < 0:
            raise ValueError("Invalid training resource event")
        self.optimizer_steps += 1
        self.examples += batch_size
        self.training_seconds += elapsed

    def as_dict(self) -> dict:
        return {**asdict(self), "online_seconds": self.online_seconds}
