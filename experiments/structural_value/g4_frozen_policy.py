"""Load only the frozen G3 full-training Ridge model for new-seed inference."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .g4_contract import G2_SOURCE_SHA, G3_SOURCE_RUN
from .scripts.value_predictor import (
    FEATURE_NAMES, TARGET_ACTIONS, RIDGE_ALPHA, STATE_KEYS, ValueModel, decide,
)


def _json_sha(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenPolicy:
    model: ValueModel
    model_sha256: str
    artifact_run_id: int

    def action(self, state: dict, *, dataset: str) -> tuple[str, dict[str, float]]:
        if set(state) != set(STATE_KEYS):
            raise AssertionError("Decision features have changed from original G3")
        if dataset not in ("Spiral", "SpiralHard"):
            raise ValueError("Unknown prospective task")
        x = np.array([state[k] for k in STATE_KEYS] +
                     [float(dataset == "SpiralHard")], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Invalid decision-time features")
        z = (x - self.model.mean) / self.model.std
        estimates = (np.r_[1.0, z] @ self.model.beta).tolist()
        if len(estimates) != len(TARGET_ACTIONS):
            raise AssertionError("Incorrect frozen target count")
        return decide(estimates), {
            name: float(value) for name, value in zip(TARGET_ACTIONS, estimates)
        }


def load_frozen_policy(path: Path) -> FrozenPolicy:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if (obj.get("schema") != "awn_structural_value_predictor_groupheldout_v1"
            or obj.get("source_grid_sha") != G2_SOURCE_SHA
            or obj.get("source_grid_run") != 44848340275
            or obj.get("training_decision_count") != 24
            or obj.get("fixed_alpha") != RIDGE_ALPHA
            or not obj.get("zero_abstention_threshold")
            or obj.get("target_actions") != list(TARGET_ACTIONS)
            or obj.get("feature_names") != list(FEATURE_NAMES)
            or obj.get("test_examples_used") != 0
            or obj.get("passed") is not True):
        raise AssertionError("G3 locked artifact provenance, policy or source invalid")
    candidate = obj.get("full_data_candidate_not_independently_validated")
    if not isinstance(candidate, dict):
        raise AssertionError("G3 final candidate missing")
    expected = {"type", "alpha_fixed", "feature_names", "target_actions",
                "mean", "std", "beta"}
    if (set(candidate) != expected
            or candidate["type"] != "train_only_standardized_multioutput_ridge"
            or candidate["alpha_fixed"] != RIDGE_ALPHA
            or candidate["feature_names"] != list(FEATURE_NAMES)
            or candidate["target_actions"] != list(TARGET_ACTIONS)):
        raise AssertionError("Unexpected model class, shape or feature contract")
    mean = np.asarray(candidate["mean"], dtype=np.float64)
    std = np.asarray(candidate["std"], dtype=np.float64)
    beta = np.asarray(candidate["beta"], dtype=np.float64)
    if (mean.shape != (10,) or std.shape != (10,) or beta.shape != (11, 3)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or not np.isfinite(beta).all() or np.any(std <= 0)):
        raise AssertionError("G3 model parameters corrupted")
    model = ValueModel(mean, std, beta)
    return FrozenPolicy(model, _json_sha(candidate), G3_SOURCE_RUN)


def verify_precommitted_prediction(
        policy: FrozenPolicy, *, state: dict, dataset: str,
        recorded_action: str, recorded_values: dict,
        absolute_tolerance: float = 1e-12) -> None:
    """Only float64 serialization/reduction noise is tolerated, not action drift.

    The chosen action must match EXACTLY. Tolerance is 10^(-12) net
    CE units, orders of magnitude below the declared G4 effect scale.
    """
    expected_action, expected_values = policy.action(state, dataset=dataset)
    if recorded_action != expected_action:
        raise AssertionError(
            f"Precommitted action changed: {recorded_action} vs {expected_action}")
    if (not isinstance(recorded_values, dict)
            or set(recorded_values) != set(expected_values)):
        raise AssertionError("Precommitted predicted target keys changed")
    for action, expected in expected_values.items():
        actual = recorded_values[action]
        if (isinstance(actual, bool) or not isinstance(actual, (float, int))
                or not np.isfinite(actual)
                or abs(float(actual) - expected) > absolute_tolerance):
            raise AssertionError(
                f"Precommitted predicted number changed beyond numerical tolerance: {action}")
