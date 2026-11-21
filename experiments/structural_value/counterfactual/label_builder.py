"""Controlled action-vs-skip labels, paired by full checkpoint/stream identity."""
from __future__ import annotations

from dataclasses import dataclass
from .fork_runner import RolloutObservation
from ..actions.action_spec import ActionKind


@dataclass(frozen=True)
class PairedLabel:
    pair_key: tuple
    action_id: str
    delta_risk: float
    delta_online_seconds: float
    net_value: float
    no_op: bool

    def as_dict(self) -> dict:
        return {
            "run_id": self.pair_key[0], "checkpoint_id": self.pair_key[1],
            "rollout_seed": self.pair_key[2], "horizon_steps": self.pair_key[3],
            "action_id": self.action_id, "delta_risk": self.delta_risk,
            "delta_online_seconds": self.delta_online_seconds,
            "net_value": self.net_value, "no_op": self.no_op,
        }


def build_labels(rows: list[RolloutObservation], *,
                 cost_weight: float, reference_seconds: float) -> list[PairedLabel]:
    if cost_weight < 0 or reference_seconds <= 0:
        raise ValueError("Invalid declared cost exchange rate")
    grouped: dict[tuple, list[RolloutObservation]] = {}
    for row in rows:
        grouped.setdefault(row.pair_key, []).append(row)
    labels = []
    for key, group in grouped.items():
        skips = [r for r in group if r.action.kind is ActionKind.SKIP]
        if len(skips) != 1:
            raise ValueError("Exactly one matched skip required per pair key")
        skip = skips[0]
        observed = set()
        for row in group:
            if row.action.action_id in observed:
                raise ValueError("Duplicate action observation")
            observed.add(row.action.action_id)
            if row.cost["optimizer_steps"] != skip.cost["optimizer_steps"] or row.cost["examples"] != skip.cost["examples"]:
                raise ValueError("Unmatched optimizer-step or example budget")
            if row.action.kind is ActionKind.SKIP:
                gain, cost = 0.0, 0.0
            else:
                gain = skip.validation_risk - row.validation_risk
                cost = row.cost["online_seconds"] - skip.cost["online_seconds"]
            labels.append(PairedLabel(
                key, row.action.action_id, gain, cost,
                gain - cost_weight * cost / reference_seconds, row.no_op))
    return labels
