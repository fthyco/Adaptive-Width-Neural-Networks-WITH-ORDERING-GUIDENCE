"""Run real intermediate-horizon endpoints from independent checkpoint forks."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol, Sequence

import numpy as np
import random
import torch

from ..actions.action_spec import ActionMeasurement, ActionSpec, ActionKind
from ..accounting.cost_ledger import OnlineCostLedger
from ..state.snapshot import ForkState, TrainingSnapshot


class RolloutAdapter(Protocol):
    device: torch.device

    def scope(self, gate_mode: str) -> AbstractContextManager: ...
    def apply(self, action: ActionSpec) -> ActionMeasurement: ...
    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None: ...
    def validation_risk(self) -> float: ...
    def widths(self) -> tuple[int, ...]: ...


@dataclass(frozen=True)
class RolloutObservation:
    run_id: str
    checkpoint_id: str
    action: ActionSpec
    rollout_seed: int
    horizon_steps: int
    validation_risk: float
    immediate_risk_before: float
    immediate_risk_after: float
    cost: dict
    final_widths: tuple[int, ...]
    no_op: bool

    @property
    def pair_key(self) -> tuple:
        return (self.run_id, self.checkpoint_id,
                self.rollout_seed, self.horizon_steps)

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id, "checkpoint_id": self.checkpoint_id,
            "action": self.action.payload, "action_id": self.action.action_id,
            "rollout_seed": self.rollout_seed,
            "horizon_steps": self.horizon_steps,
            "validation_risk": self.validation_risk,
            "immediate_risk_before": self.immediate_risk_before,
            "immediate_risk_after": self.immediate_risk_after,
            "cost": self.cost, "final_widths": list(self.final_widths),
            "no_op": self.no_op,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time(device: torch.device, operation: Callable[[], object]) -> tuple[object, float]:
    _sync(device)
    started = perf_counter()
    output = operation()
    _sync(device)
    return output, perf_counter() - started


def paired_rollouts(
    snapshot: TrainingSnapshot,
    adapter_factory: Callable[[ForkState], RolloutAdapter],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    actions: Sequence[ActionSpec],
    horizons: Sequence[int],
    *,
    run_id: str,
    checkpoint_id: str,
    rollout_seed: int,
) -> list[RolloutObservation]:
    """Each branch restores model, optimizer, sampler/RNG and sees identical batches.

    A whole continuation is run through max(H); every smaller H is measured
    at the actual intermediate endpoint before training moves on.
    """
    hset = set(horizons)
    if not hset or min(hset) <= 0 or max(hset) > len(batches):
        raise ValueError("Horizons must be positive optimizer steps covered by batches")
    if not run_id or not checkpoint_id:
        raise ValueError("Missing experiment identity")
    if not any(a.kind is ActionKind.SKIP for a in actions):
        raise ValueError("A skip branch is mandatory")
    if len({a.action_id for a in actions}) != len(actions):
        raise ValueError("Duplicate action payloads")
    rows = []
    for action in actions:
        restored = snapshot.fork()
        # This explicit future-stream seed is applied equally to each branch.
        random.seed(rollout_seed)
        np.random.seed(rollout_seed % (2**32))
        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
        adapter = adapter_factory(restored)
        ledger = OnlineCostLedger()
        with adapter.scope(action.gate_mode):
            before, duration = _time(adapter.device, adapter.validation_risk)
            ledger.evaluation_seconds += duration
            effect = adapter.apply(action)
            ledger.evidence_seconds += effect.evidence_seconds
            ledger.intervention_seconds += effect.intervention_seconds
            after, duration = _time(adapter.device, adapter.validation_risk)
            ledger.evaluation_seconds += duration
            for index, (x, y) in enumerate(batches[:max(hset)], 1):
                _, duration = _time(adapter.device,
                                    lambda x=x, y=y: adapter.train_step((x, y)))
                ledger.step(len(x), duration)
                if index in hset:
                    risk, duration = _time(adapter.device, adapter.validation_risk)
                    ledger.evaluation_seconds += duration
                    if not np.isfinite(risk):
                        raise ValueError("Nonfinite validation risk")
                    rows.append(RolloutObservation(
                        run_id, checkpoint_id, action, rollout_seed, index, float(risk),
                        float(before), float(after), ledger.as_dict(),
                        adapter.widths(), effect.no_op))
    return rows
