"""Fail-closed tests for causal-value targets, grouped holdouts, and train-only preprocessing."""
from __future__ import annotations

from copy import deepcopy
import math

import pytest
import numpy as np

from experiments.structural_value.grid_contract import ACTIONS, CHECKPOINTS, DATASETS, SEEDS
from experiments.structural_value.scripts.value_predictor import (
    SOURCE_GRID_SHA, FEATURE_NAMES, RIDGE_ALPHA,
    _one_fold, build_decisions, decide, evaluate_group_heldout,
    fit_value_model,
)


def _valid_source_reports():
    reports = []
    for dataset in DATASETS:
        for seed in SEEDS:
            cps = []
            for epoch in CHECKPOINTS:
                cp = f"{dataset}:seed{seed}:epoch{epoch}"
                baseline = 0.6 + epoch * 0.0001
                state = {
                    "epoch": epoch, "physical_width": 48 + epoch,
                    "learned_width": 48 + epoch, "width_rate": 0.01 + epoch * 0.0001,
                    "train_elbo_loss": 0.8 - epoch * 0.005,
                    "validation_ce": baseline,
                    "validation_accuracy": 0.5 + epoch * 0.001,
                    "validation_ce_change_last_five_epochs": -0.001 * epoch,
                    "training_steps_so_far": epoch * (40 if dataset == "Spiral" else 80),
                }
                rows = []
                vals = {
                    "skip": 0,
                    "random_positional": -0.05 + 0.0001 * seed,
                    "weight_positional": 0.012 if epoch in (5, 10) else -0.01,
                    "gradient_positional": (
                        0.008 if dataset == "Spiral" and epoch == 20 else -0.006
                    ),
                }
                for action in ACTIONS:
                    gain = vals[action]
                    rows.append({
                        "action": action, "gate_mode": "positional",
                        "validation_ce": baseline - gain,
                        "delta_risk": gain,
                        "delta_online_seconds": 0.0,
                        "net_value": gain,
                        "no_op": action == "skip",
                        "final_widths": [48 + epoch],
                        "optimizer_steps": 8, "training_examples": 1024,
                    })
                cps.append({
                    "epoch": epoch, "checkpoint_id": cp,
                    "decision_state": state, "final_horizon_comparison": rows,
                })
            reports.append({
                "schema": "awn_structural_value_grid96_per_run_audit_v1",
                "dataset": dataset, "seed": seed, "git_sha": SOURCE_GRID_SHA,
                "passed": True, "test_examples_used": 0,
                "checkpoint_epochs": list(CHECKPOINTS),
                "final_horizon_rollouts": 16, "all_horizon_observations": 48,
                "split": {"test_examples_used": 0},
                "checkpoints": cps,
            })
    return reports


def test_build_exactly_24_decisions_and_preaction_feature_contract():
    decisions = build_decisions(_valid_source_reports())
    assert len(decisions) == 24
    assert len(set(d.checkpoint_id for d in decisions)) == 24
    assert len(FEATURE_NAMES) == 10
    assert all(len(d.x) == len(FEATURE_NAMES) for d in decisions)
    assert all(set(d.outcomes) == set(ACTIONS) for d in decisions)
    assert all(d.outcomes["skip"]["net_value"] == 0 for d in decisions)


def test_fixed_ridge_is_deterministic_and_no_unseen_seed_feature():
    ds = build_decisions(_valid_source_reports())
    train = [d for d in ds if d.seed != 42]
    test = [d for d in ds if d.seed == 42]
    model = fit_value_model(train)
    assert model.beta.shape == (len(FEATURE_NAMES) + 1, 3)
    assert model.as_dict()["alpha_fixed"] == RIDGE_ALPHA
    assert "seed" not in model.as_dict()["feature_names"]
    assert model.predict(test).shape == (8, 3)
    assert np.array_equal(model.predict(test), fit_value_model(train).predict(test))


def test_zero_is_abstention_and_action_order_is_explicit():
    assert decide([-0.01, -0.02, -0.03]) == "skip"
    assert decide([0.0, 0.0, 0.0]) == "skip"
    assert decide([0.1, 0.07, -0.1]) == "random_positional"
    assert decide([0.05, 0.11, 0.09]) == "weight_positional"
    assert decide([-0.05, -0.01, 0.002]) == "gradient_positional"
    with pytest.raises(ValueError):
        decide([0, float("nan"), 0])


def test_group_holdout_coverage_and_nondeployable_oracle_marker():
    report = evaluate_group_heldout(_valid_source_reports())
    assert report["passed"]
    assert report["test_examples_used"] == 0
    assert set(report["fold_protocols"]) == {
        "leave_one_seed_out", "leave_one_dataset_out"}
    seed = report["fold_protocols"]["leave_one_seed_out"]
    domain = report["fold_protocols"]["leave_one_dataset_out"]
    assert seed["summary"]["heldout_decisions"] == 24
    assert domain["summary"]["heldout_decisions"] == 24
    assert len(seed["folds"]) == 3
    assert len(domain["folds"]) == 2
    assert all(f["train_checkpoint_count"] == 16 and
               f["heldout_checkpoint_count"] == 8 for f in seed["folds"])
    assert all(f["train_checkpoint_count"] == 12 and
               f["heldout_checkpoint_count"] == 12 for f in domain["folds"])
    assert seed["summary"]["policies"]["always_skip"]["mean_net_value"] == 0
    assert "hindsight_oracle_upper_bound" in seed["summary"]["policies"]
    for protocol in report["fold_protocols"].values():
        assert all(not (
            {tuple(t) for t in fold["train_trajectories"]}
            & {tuple(t) for t in fold["heldout_trajectories"]}
        ) for fold in protocol["folds"])


def test_heldout_labels_cannot_affect_heldout_predictions():
    source = _valid_source_reports()
    original = evaluate_group_heldout(source)
    changed = deepcopy(source)
    for report in changed:
        if report["seed"] != 42:
            continue
        for cp in report["checkpoints"]:
            for row in cp["final_horizon_comparison"]:
                if row["action"] == "weight_positional":
                    row["net_value"] += 0.2
                    row["delta_risk"] += 0.2
                    row["validation_ce"] -= 0.2
    modified = evaluate_group_heldout(changed)
    def fold(x):
        return next(f for f in x["fold_protocols"]["leave_one_seed_out"]["folds"]
                    if f["fold"] == "heldout_seed_42")
    first, last = fold(original), fold(modified)
    assert [d["predicted_action"] for d in first["decisions"]] == [
        d["predicted_action"] for d in last["decisions"]]
    assert [d["predicted_net_value"] for d in first["decisions"]] == [
        d["predicted_net_value"] for d in last["decisions"]]
    assert first["fitted_model"] == last["fitted_model"]


@pytest.mark.parametrize("corruption", [
    "nonfinite_label", "invalid_source_commit", "duplicate_checkpoint",
    "missing_action", "postdecision_feature", "test_leakage",
    "invalid_skip", "risk_pair_error", "missing_group",
])
def test_invalid_or_leaky_teacher_inputs_fail_closed(corruption):
    reports = _valid_source_reports()
    target = reports[0]
    cp = target["checkpoints"][0]
    if corruption == "nonfinite_label":
        cp["final_horizon_comparison"][1]["net_value"] = float("nan")
    elif corruption == "invalid_source_commit":
        target["git_sha"] = "0" * 40
    elif corruption == "duplicate_checkpoint":
        target["checkpoints"][1]["checkpoint_id"] = cp["checkpoint_id"]
    elif corruption == "missing_action":
        cp["final_horizon_comparison"].pop()
    elif corruption == "postdecision_feature":
        cp["decision_state"]["future_gain"] = 1.0
    elif corruption == "test_leakage":
        target["test_examples_used"] = 1
    elif corruption == "invalid_skip":
        cp["final_horizon_comparison"][0]["net_value"] = 0.2
    elif corruption == "risk_pair_error":
        cp["final_horizon_comparison"][1]["delta_risk"] = -0.15
    elif corruption == "missing_group":
        reports.pop()
    with pytest.raises((AssertionError, ValueError, KeyError)):
        evaluate_group_heldout(reports)


def test_checkpoint_trajectory_overlap_is_rejected():
    ds = build_decisions(_valid_source_reports())
    with pytest.raises(AssertionError, match="trajectory leaked"):
        _one_fold("leaky", ds[:18], ds[16:])
