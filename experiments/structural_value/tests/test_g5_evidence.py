"""G5A fail-closed tests; no native AWN runtime needed."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.structural_value.g5_evidence_features import (
    FEATURE_NAMES, evidence_order_and_concentration, order_features,
    proposal_features)
from experiments.structural_value.grid_contract import DATASETS, SEEDS, CHECKPOINTS
from experiments.structural_value.scripts.g5_evidence_ablation import (
    COMBINED_NAMES, JoinedDecision, fold_result, fit_augmented, join_features,
    predict_augmented)
from experiments.structural_value.scripts.value_predictor import (
    build_decisions, SOURCE_GRID_SHA)
from experiments.structural_value.tests.test_value_predictor import _valid_source_reports


def test_proposal_metrics_no_modification_and_bijective_validation():
    w = np.array([2., 4., 1., 3.])
    g = np.array([1., 3., 4., 2.])
    initial_w, initial_g = w.copy(), g.copy()
    x = proposal_features(w_scores=w, g_scores=g, learned_width=2,
                          random_action_seed=1234)
    assert list(x) == list(FEATURE_NAMES)
    assert all(0 <= value <= 1 for value in x.values())
    assert x["weight_prefix_replacement"] == 0.5
    assert x["gradient_prefix_replacement"] == 0.5
    assert x == proposal_features(w_scores=w, g_scores=g, learned_width=2,
                                  random_action_seed=1234)
    assert np.array_equal(w, initial_w) and np.array_equal(g, initial_g)
    assert order_features([0, 1, 2, 3], n=4, learned_width=3) == (0.0, 0.0)


@pytest.mark.parametrize("scores", [[float("nan")], [-1], [float("inf")], [[1, 2]], []])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        evidence_order_and_concentration(scores)


@pytest.mark.parametrize("order", [[1, 1, 2], [0, 1], [0., 2., 1.]])
def test_invalid_permutation_rejected(order):
    with pytest.raises(ValueError):
        order_features(order, n=3, learned_width=2)


def test_full_prefix_has_zero_replacement_and_identical_weights_have_deterministic_tie():
    x = proposal_features(w_scores=np.ones(5), g_scores=np.ones(5),
                          learned_width=5, random_action_seed=4)
    assert x["random_prefix_replacement"] == 0
    assert x["weight_prefix_replacement"] == 0
    assert x["gradient_prefix_replacement"] == 0
    assert x["weight_mean_rank_displacement"] == 0


def _synthetic_joined():
    data = build_decisions(_valid_source_reports())
    return [
        JoinedDecision(d, tuple(0.01 * (j + 1) * (1 + d.epoch / 50)
                                for j in range(len(FEATURE_NAMES))))
        for d in data
    ]


def test_augmented_ridge_has_fixed_regularization_and_heldout_is_unseen():
    ds = _synthetic_joined()
    train = [d for d in ds if d.base.seed != 42]
    test = [d for d in ds if d.base.seed == 42]
    model = fit_augmented(train)
    assert model["alpha_fixed"] == 12
    assert len(model["feature_names"]) == len(COMBINED_NAMES) == 18
    assert model["target_actions"] == [
        "random_positional", "weight_positional", "gradient_positional"
    ]
    assert predict_augmented(model, test).shape == (8, 3)
    assert not set(x.trajectory for x in train) & set(x.trajectory for x in test)
    assert fold_result("test", train, test)["evaluation_count"] == 8


def test_unseen_outcomes_do_not_affect_predictions():
    ds = _synthetic_joined()
    train = [d for d in ds if d.base.seed != 42]
    heldout = [d for d in ds if d.base.seed == 42]
    before = predict_augmented(fit_augmented(train), heldout)
    changed = deepcopy(heldout)
    for d in changed:
        d.base.outcomes["weight_positional"]["net_value"] += 100
    after = predict_augmented(fit_augmented(train), changed)
    assert np.array_equal(before, after)


def _write_feature_replay(root, *, tamper=None):
    reports = _valid_source_reports()
    dmap = {(d.dataset, d.seed, d.epoch): d for d in build_decisions(reports)}
    for ds in DATASETS:
        for seed in SEEDS:
            cps = []
            for epoch in CHECKPOINTS:
                d = dmap[(ds, seed, epoch)]
                cps.append({
                    "dataset": ds, "seed": seed, "epoch": epoch,
                    "checkpoint_id": d.checkpoint_id,
                    "reference_decision_state": dict(zip((
                        "epoch", "physical_width", "learned_width", "width_rate",
                        "train_elbo_loss", "validation_ce", "validation_accuracy",
                        "validation_ce_change_last_five_epochs",
                        "training_steps_so_far"), d.x[:9])),
                    "proposal_features": dict.fromkeys(FEATURE_NAMES, 0.25),
                    "no_intervention_applied": True,
                })
            obj = {
                "schema": "awn_g5_predecision_evidence_replay_v1",
                "source_g2_commit": SOURCE_GRID_SHA,
                "source_g2_run": 44848340275,
                "feature_extraction_commit": "a" * 40,
                "dataset": ds, "seed": seed,
                "feature_names": list(FEATURE_NAMES),
                "checkpoints": cps,
                "test_examples_used": 0,
                "future_training_steps_executed": 0,
                "interventions_applied": 0,
                "feature_source": "reference_native_awn_before_action_W_G_and_seeded_random",
            }
            if tamper and (ds, seed) == ("Spiral", 42):
                tamper(obj)
            p = root / f"g5_{ds}_seed{seed}"
            p.mkdir()
            (p / "features.json").write_text(json.dumps(obj))
    return reports


def test_replayed_features_exactly_join_24_historical_decisions(tmp_path):
    reports = _write_feature_replay(tmp_path)
    rows, sha = join_features(reports, tmp_path)
    assert len(rows) == 24 and sha == "a" * 40
    assert all(len(d.vector) == 18 for d in rows)


@pytest.mark.parametrize("tamper", [
    lambda o: o["checkpoints"][0]["proposal_features"].update({"future_value": 1}),
    lambda o: o["checkpoints"][0].update({"checkpoint_id": "wrong"}),
    lambda o: o["checkpoints"][0]["reference_decision_state"].update({"validation_ce": 99}),
    lambda o: o.update({"interventions_applied": 1}),
    lambda o: o.update({"test_examples_used": 1}),
    lambda o: o["checkpoints"][0].update({"no_intervention_applied": False}),
])
def test_replay_must_match_checkpoint_and_never_contain_future_labels(tmp_path, tamper):
    reports = _write_feature_replay(tmp_path, tamper=tamper)
    with pytest.raises(AssertionError):
        join_features(reports, tmp_path)
