"""G4 frozen-model and future-seed contracts without native AWN dependencies."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from experiments.structural_value.g4_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, HORIZON_STEPS, ACTIONS,
    G3_SOURCE_RUN, G3_SOURCE_ARTIFACT_ID,
)
from experiments.structural_value.g4_frozen_policy import (
    load_frozen_policy, _json_sha,
)
from experiments.structural_value.scripts.value_predictor import (
    FEATURE_NAMES, STATE_KEYS, RIDGE_ALPHA, TARGET_ACTIONS, SOURCE_GRID_SHA,
)


def _g3_payload():
    beta = [[0.01, -0.01, -0.02]] + [[0.0, 0.0, 0.0] for _ in range(10)]
    candidate = {
        "type": "train_only_standardized_multioutput_ridge",
        "alpha_fixed": 12.0,
        "feature_names": list(FEATURE_NAMES),
        "target_actions": list(TARGET_ACTIONS),
        "mean": [0.0] * 10,
        "std": [1.0] * 10,
        "beta": beta,
    }
    return {
        "schema": "awn_structural_value_predictor_groupheldout_v1",
        "source_grid_sha": SOURCE_GRID_SHA,
        "source_grid_run": 44848340275,
        "training_decision_count": 24,
        "fixed_alpha": RIDGE_ALPHA,
        "zero_abstention_threshold": True,
        "target_actions": list(TARGET_ACTIONS),
        "feature_names": list(FEATURE_NAMES),
        "test_examples_used": 0,
        "passed": True,
        "full_data_candidate_not_independently_validated": candidate,
    }


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _valid_state():
    return {
        "epoch": 5, "physical_width": 256, "learned_width": 255,
        "width_rate": 0.01, "train_elbo_loss": 0.6,
        "validation_ce": 0.5, "validation_accuracy": 0.5,
        "validation_ce_change_last_five_epochs": 0.01,
        "training_steps_so_far": 128,
    }


def test_future_seeds_are_frozen_and_disjoint_from_prior_grid():
    assert SEEDS == (20271, 20272, 20273)
    assert not (set(SEEDS) & {42, 123, 456, 789, 1000})
    assert DATASETS == ("Spiral", "SpiralHard")
    assert CHECKPOINTS == (5, 10, 20, 35)
    assert HORIZON_STEPS == 8 and HORIZONS == (1, 4, 8)
    assert ACTIONS == ("skip", "random_positional", "weight_positional",
                       "gradient_positional")
    assert G3_SOURCE_RUN == 44849364139
    assert G3_SOURCE_ARTIFACT_ID == 14683998134


def test_policy_is_loaded_from_valid_old_artifact_only(tmp_path):
    original = _g3_payload()
    policy = load_frozen_policy(_write(tmp_path / "g3.json", original))
    assert policy.model_sha256 == _json_sha(original["full_data_candidate_not_independently_validated"])
    assert policy.artifact_run_id == G3_SOURCE_RUN
    action, prediction = policy.action(_valid_state(), dataset="Spiral")
    assert action == "random_positional"
    assert prediction["random_positional"] == pytest.approx(0.01)
    assert prediction["weight_positional"] == pytest.approx(-0.01)
    assert prediction["gradient_positional"] == pytest.approx(-0.02)


def test_predecision_inputs_cannot_include_future_information(tmp_path):
    policy = load_frozen_policy(_write(tmp_path / "g3.json", _g3_payload()))
    state = _valid_state()
    state["future_delta_risk"] = 0.5
    with pytest.raises(AssertionError):
        policy.action(state, dataset="Spiral")
    state.pop("future_delta_risk")
    state.pop("validation_ce")
    with pytest.raises(AssertionError):
        policy.action(state, dataset="Spiral")
    with pytest.raises(ValueError):
        policy.action(_valid_state(), dataset="UnseenTask")


@pytest.mark.parametrize("corruption", [
    "source_sha", "source_run", "size", "threshold", "missing_candidate",
    "model_type", "extra_model_key", "feature_order", "target_order",
    "nonfinite_beta", "nonpositive_std", "bad_shape", "test_label_leak",
])
def test_reject_changed_or_incomplete_historical_policy(tmp_path, corruption):
    x = _g3_payload()
    c = x["full_data_candidate_not_independently_validated"]
    if corruption == "source_sha":
        x["source_grid_sha"] = "b" * 40
    elif corruption == "source_run":
        x["source_grid_run"] = 99
    elif corruption == "size":
        x["training_decision_count"] = 23
    elif corruption == "threshold":
        x["zero_abstention_threshold"] = False
    elif corruption == "missing_candidate":
        del x["full_data_candidate_not_independently_validated"]
    elif corruption == "model_type":
        c["type"] = "unpredeclared_deep_network"
    elif corruption == "extra_model_key":
        c["test_reward"] = 0.1
    elif corruption == "feature_order":
        c["feature_names"] = list(reversed(c["feature_names"]))
    elif corruption == "target_order":
        c["target_actions"] = list(reversed(c["target_actions"]))
    elif corruption == "nonfinite_beta":
        c["beta"][0][0] = float("nan")
    elif corruption == "nonpositive_std":
        c["std"][3] = -1.0
    elif corruption == "bad_shape":
        c["beta"].pop()
    elif corruption == "test_label_leak":
        x["test_examples_used"] = 10
    with pytest.raises((AssertionError, ValueError)):
        load_frozen_policy(_write(tmp_path / "g3.json", x))


def test_policy_action_is_independent_of_any_new_seed_or_observed_label(tmp_path):
    policy = load_frozen_policy(_write(tmp_path / "g3.json", _g3_payload()))
    state = _valid_state()
    first = policy.action(state, dataset="SpiralHard")
    second = policy.action(dict(state), dataset="SpiralHard")
    assert first == second
    assert set(first[1]) == set(TARGET_ACTIONS)
    assert set(state) == set(STATE_KEYS)
