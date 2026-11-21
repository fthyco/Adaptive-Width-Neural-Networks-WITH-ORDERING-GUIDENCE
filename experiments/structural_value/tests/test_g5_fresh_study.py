"""Fresh-cohort contract tests, no PyTorch runtime required."""
from dataclasses import replace
import json

from experiments.structural_value.g5_fresh_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, HORIZON_STEPS)
from experiments.structural_value.g5_evidence_features import FEATURE_NAMES
from experiments.structural_value.scripts.g5_evidence_ablation import JoinedDecision
from experiments.structural_value.scripts.g5_fresh_ablation import evaluate
from experiments.structural_value.scripts.value_predictor import (
    Decision, build_decisions)
from experiments.structural_value.tests.test_value_predictor import _valid_source_reports


def test_fresh_seeds_and_horizon_separately_declared():
    assert DATASETS == ("Spiral", "SpiralHard")
    assert SEEDS == (20331, 20332, 20333)
    assert not (set(SEEDS) & {42, 123, 456, 20271, 20272, 20273})
    assert CHECKPOINTS == (5, 10, 20, 35)
    assert HORIZONS == (1, 4, 8) and HORIZON_STEPS == 8
    assert len(FEATURE_NAMES) == 8


def test_new_seed_group_holdouts_compare_same_training_and_heldout_units():
    original = build_decisions(_valid_source_reports())
    mapping = {42: SEEDS[0], 123: SEEDS[1], 456: SEEDS[2]}
    cohorts = []
    for d in original:
        seed = mapping[d.seed]
        nd = replace(d, seed=seed, checkpoint_id=f"{d.dataset}:seed{seed}:epoch{d.epoch}")
        f = tuple(0.01 + d.epoch / (i + 1000) for i in range(8))
        cohorts.append(JoinedDecision(nd, f))
    result = evaluate(cohorts, "a" * 40)
    assert result["passed"]
    assert result["decisions"] == 24 and result["paired_final_horizon_rollouts"] == 96
    assert result["test_examples_used"] == 0
    json.dumps(result, allow_nan=False)  # Fail before publishing an invalid NaN aggregate
    assert len(result["seed_holdout"]["folds"]) == 3
    assert len(result["dataset_holdout"]["folds"]) == 2
    for fold in result["seed_holdout"]["folds"] + result["dataset_holdout"]["folds"]:
        assert not (
            {tuple(x) for x in fold["train_trajectories"]}
            & {tuple(x) for x in fold["heldout_trajectories"]}
        )
        assert fold["training_count"] + fold["evaluation_count"] == 24
