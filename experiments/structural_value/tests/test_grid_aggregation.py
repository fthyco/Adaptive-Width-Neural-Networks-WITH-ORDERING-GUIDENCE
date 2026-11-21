"""Independent grid-shape and no-leakage checks before long AWN CI."""
from copy import deepcopy

import pytest

from experiments.structural_value.scripts.aggregate_grid import aggregate_reports
from experiments.structural_value.scripts.run_grid import (
    CHECKPOINTS, DATASETS, SEEDS, HORIZON_STEPS)


def _valid_reports():
    reports = []
    for ds in DATASETS:
        for seed in SEEDS:
            checkpoints = []
            for epoch in CHECKPOINTS:
                rows = []
                for name in ("skip", "random_positional",
                             "weight_positional", "gradient_positional"):
                    rows.append({
                        "action": name, "gate_mode": "positional",
                        "no_op": name == "skip",
                        "validation_ce": 0.5, "delta_risk": 0.0 if name == "skip" else 0.01,
                        "delta_online_seconds": 0.0,
                        "net_value": 0.0 if name == "skip" else 0.01,
                        "final_widths": [16], "optimizer_steps": HORIZON_STEPS,
                        "training_examples": 8 * 128,
                    })
                checkpoints.append({
                    "epoch": epoch, "checkpoint_id": f"{ds}:{seed}:{epoch}",
                    "final_horizon_comparison": rows,
                })
            reports.append({
                "schema": "awn_structural_value_grid96_per_run_audit_v1",
                "dataset": ds, "seed": seed, "git_sha": "1" * 40,
                "passed": True, "test_examples_used": 0,
                "checkpoint_epochs": list(CHECKPOINTS),
                "final_horizon_rollouts": 16,
                "all_horizon_observations": 48,
                "split": {"test_examples_used": 0},
                "checkpoints": checkpoints,
            })
    return reports


def test_full_shape_and_seed_cluster_descriptives():
    r = aggregate_reports(_valid_reports())
    assert r["passed"]
    assert r["dataset_seed_runs"] == 6
    assert r["checkpoint_decisions"] == 24
    assert r["final_horizon_rollouts"] == 96
    assert r["all_horizon_observations"] == 288
    assert len(r["descriptive_summaries"]) == 8
    x = next(x for x in r["descriptive_summaries"] if
             x["dataset"] == "Spiral" and x["action"] == "weight_positional")
    assert x["mean_delta_risk"] == pytest.approx(0.01)
    assert x["effective_interventions"] == 12
    assert x["seed_cluster_sd_delta_risk"] == 0


@pytest.mark.parametrize("error", ["missing", "duplicate", "test", "checkpoint",
                                   "budget", "actions", "commit", "nan"])
def test_aggregate_rejects_corrupted_or_incomplete_grid(error):
    items = _valid_reports()
    if error == "missing":
        items.pop()
    elif error == "duplicate":
        items[1]["seed"] = items[0]["seed"]
    elif error == "test":
        items[0]["test_examples_used"] = 1
    elif error == "checkpoint":
        items[0]["checkpoints"][0]["epoch"] = 99
    elif error == "budget":
        items[0]["checkpoints"][0]["final_horizon_comparison"][1]["training_examples"] = 12
    elif error == "actions":
        items[0]["checkpoints"][0]["final_horizon_comparison"][2]["action"] = "skip"
    elif error == "commit":
        items[0]["git_sha"] = "2" * 40
    elif error == "nan":
        items[0]["checkpoints"][0]["final_horizon_comparison"][1]["delta_risk"] = float("nan")
    with pytest.raises((AssertionError, ValueError)):
        aggregate_reports(items)
