"""Exploratory paired G3-vs-G5A predictor ablation on immutable historical G2.

All action outcomes are read only in this evaluation stage, never in the
G5A predecision feature replay. Both predictors use the identical outer
folds, labels, regularization and zero abstention threshold. This is NOT
an independent confirmation and MUST NOT be confused with frozen G4.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from ..g5_evidence_features import FEATURE_NAMES as PROPOSAL_NAMES
from ..grid_contract import ACTIONS, CHECKPOINTS, DATASETS, SEEDS
from .value_predictor import (
    Decision, FEATURE_NAMES as BASE_NAMES, STATE_KEYS, TARGET_ACTIONS,
    RIDGE_ALPHA, SOURCE_GRID_SHA, build_decisions, decide, fit_value_model,
    load_reports,
)

COMBINED_NAMES = BASE_NAMES + PROPOSAL_NAMES
SOURCE_G2_RUN = 44848340275


@dataclass(frozen=True)
class JoinedDecision:
    base: Decision
    proposal: tuple[float, ...]

    @property
    def trajectory(self):
        return self.base.trajectory

    @property
    def vector(self):
        return self.base.x + self.proposal


def join_features(reports: list[dict], root: Path) -> tuple[list[JoinedDecision], str]:
    historical = build_decisions(reports)
    original = {(d.dataset, d.seed, d.epoch): d for d in historical}
    lookup = {}
    commits = set()
    for dataset in DATASETS:
        for seed in SEEDS:
            path = Path(root) / f"g5_{dataset}_seed{seed}" / "features.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            if (set(raw) != {
                "schema", "source_g2_commit", "source_g2_run",
                "feature_extraction_commit", "dataset", "seed", "feature_names",
                "checkpoints", "test_examples_used",
                "future_training_steps_executed", "interventions_applied",
                "feature_source"
            } or raw["schema"] != "awn_g5_predecision_evidence_replay_v1"
                    or raw["dataset"] != dataset or raw["seed"] != seed
                    or raw["source_g2_commit"] != SOURCE_GRID_SHA
                    or raw["source_g2_run"] != SOURCE_G2_RUN
                    or raw["feature_names"] != list(PROPOSAL_NAMES)
                    or raw["test_examples_used"] != 0
                    or raw["future_training_steps_executed"] != 0
                    or raw["interventions_applied"] != 0
                    or raw["feature_source"] !=
                    "reference_native_awn_before_action_W_G_and_seeded_random"
                    or len(raw["checkpoints"]) != len(CHECKPOINTS)):
                raise AssertionError("Unapproved G5A predecision feature artifact")
            commit = raw["feature_extraction_commit"]
            if len(commit) != 40 or not all(c in "0123456789abcdef" for c in commit):
                raise AssertionError("Feature provenance lacks a Git commit")
            commits.add(commit)
            for cp in raw["checkpoints"]:
                if set(cp) != {
                    "dataset", "seed", "epoch", "checkpoint_id",
                    "reference_decision_state", "proposal_features",
                    "no_intervention_applied"
                } or cp["dataset"] != dataset or cp["seed"] != seed or (
                    cp["epoch"] not in CHECKPOINTS or cp["no_intervention_applied"] is not True
                ):
                    raise AssertionError("Invalid or non-predecision checkpoint record")
                key = (dataset, seed, cp["epoch"])
                if key in lookup or key not in original:
                    raise AssertionError("Unmatched or duplicate G5 checkpoint")
                prior = original[key]
                if cp["checkpoint_id"] != prior.checkpoint_id:
                    raise AssertionError("G5 feature checkpoint identity mismatches G2 labels")
                state = cp["reference_decision_state"]
                if set(state) != set(STATE_KEYS):
                    raise AssertionError("G5 replay did not preserve original state")
                expected = prior.x[:len(STATE_KEYS)]
                for name, number in zip(STATE_KEYS, expected):
                    if not np.isclose(state[name], number, rtol=1e-5, atol=1e-6):
                        raise AssertionError("Mismatched historical network before action")
                features = cp["proposal_features"]
                if set(features) != set(PROPOSAL_NAMES):
                    raise AssertionError("Outcome-derived or incomplete predictor inputs")
                vector = tuple(float(features[n]) for n in PROPOSAL_NAMES)
                if (not np.isfinite(vector).all()
                        or any(not 0 <= v <= 1 for v in vector)):
                    raise AssertionError("Proposal descriptor outside valid domain")
                lookup[key] = JoinedDecision(prior, vector)
    if set(lookup) != set(original) or len(commits) != 1:
        raise AssertionError("Incomplete or mixed-commit historical feature replay")
    return [lookup[(d.dataset, d.seed, d.epoch)] for d in historical], next(iter(commits))


def fit_augmented(train: list[JoinedDecision]) -> dict:
    if len(train) < 8 or len({d.base.checkpoint_id for d in train}) != len(train):
        raise ValueError("Invalid small or duplicate training set")
    x = np.asarray([d.vector for d in train], dtype=np.float64)
    y = np.asarray([[d.base.outcomes[a]["net_value"] for a in TARGET_ACTIONS]
                    for d in train], dtype=np.float64)
    if x.shape[1] != len(COMBINED_NAMES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError("Malformed predictor training data")
    mean = x.mean(0)
    std = np.where(x.std(0) <= 1e-12, 1.0, x.std(0))
    z = (x - mean) / std
    design = np.column_stack((np.ones(len(x)), z))
    regularizer = np.eye(len(COMBINED_NAMES) + 1) * RIDGE_ALPHA
    regularizer[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return {
        "type": "G5A_exploratory_multioutput_ridge",
        "alpha_fixed": RIDGE_ALPHA, "feature_names": list(COMBINED_NAMES),
        "target_actions": list(TARGET_ACTIONS),
        "mean": mean.tolist(), "std": std.tolist(), "beta": beta.tolist(),
    }


def predict_augmented(model: dict, heldout: list[JoinedDecision]) -> np.ndarray:
    x = np.asarray([d.vector for d in heldout], dtype=np.float64)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["std"])
    predicted = np.column_stack((np.ones(len(x)), z)) @ np.asarray(model["beta"])
    if predicted.shape != (len(heldout), 3) or not np.isfinite(predicted).all():
        raise AssertionError("Invalid held-out augmented predictions")
    return predicted


def fold_result(name: str, train: list[JoinedDecision], heldout: list[JoinedDecision]) -> dict:
    if not train or not heldout or set(d.trajectory for d in train) & {
        d.trajectory for d in heldout
    }:
        raise AssertionError("Leaked trajectory across evaluation fold")
    baseline = fit_value_model([d.base for d in train])
    augmented = fit_augmented(train)
    base_prediction = baseline.predict([d.base for d in heldout])
    aug_prediction = predict_augmented(augmented, heldout)
    rows = []
    for d, base_p, aug_p in zip(heldout, base_prediction, aug_prediction):
        base_action, aug_action = decide(base_p), decide(aug_p)
        outcomes = d.base.outcomes
        rows.append({
            "dataset": d.base.dataset, "seed": d.base.seed,
            "epoch": d.base.epoch, "checkpoint_id": d.base.checkpoint_id,
            "g3_action": base_action, "g5a_action": aug_action,
            "g3_observed_value": outcomes[base_action]["net_value"],
            "g5a_observed_value": outcomes[aug_action]["net_value"],
            "g3_prediction": dict(zip(TARGET_ACTIONS, map(float, base_p))),
            "g5a_prediction": dict(zip(TARGET_ACTIONS, map(float, aug_p))),
            "actual_values_diagnostic": {
                a: outcomes[a]["net_value"] for a in ACTIONS
            },
        })
    return {
        "name": name, "train_trajectories": sorted({d.trajectory for d in train}),
        "heldout_trajectories": sorted({d.trajectory for d in heldout}),
        "training_count": len(train), "evaluation_count": len(heldout),
        "rows": rows,
    }


def summarize(folds: list[dict]) -> dict:
    rows = [r for f in folds for r in f["rows"]]
    if len(rows) != 24 or len({r["checkpoint_id"] for r in rows}) != 24:
        raise AssertionError("Fold coverage must be 24 distinct checkpoints")
    out = {}
    for policy in ("g3", "g5a"):
        values = [r[f"{policy}_observed_value"] for r in rows]
        out[policy] = {
            "mean_net_value": float(np.mean(values)),
            "beneficial_decisions": sum(v > 0 for v in values),
            "harmful_decisions": sum(v < 0 for v in values),
            "worst_value": float(min(values)),
            "action_counts": {
                a: sum(r[f"{policy}_action"] == a for r in rows) for a in ACTIONS
            },
            "mae_by_action": {
                a: float(np.mean([
                    abs(r[f"{policy}_prediction"][a] -
                        r["actual_values_diagnostic"][a])
                    for r in rows]))
                for a in TARGET_ACTIONS
            },
        }
    out["difference_g5a_minus_g3"] = float(
        out["g5a"]["mean_net_value"] - out["g3"]["mean_net_value"])
    out["by_training_trajectory"] = [
        {
            "dataset": ds, "seed": seed,
            "g3_mean": float(np.mean([r["g3_observed_value"] for r in rows
                                      if r["dataset"] == ds and r["seed"] == seed])),
            "g5a_mean": float(np.mean([r["g5a_observed_value"] for r in rows
                                       if r["dataset"] == ds and r["seed"] == seed])),
        }
        for ds in DATASETS for seed in SEEDS
    ]
    return out


def evaluate(reports: list[dict], features_dir: Path) -> dict:
    joined, replay_sha = join_features(reports, features_dir)
    seed_folds = [fold_result(
        f"heldout_seed_{seed}",
        [d for d in joined if d.base.seed != seed],
        [d for d in joined if d.base.seed == seed],
    ) for seed in SEEDS]
    dataset_folds = [fold_result(
        f"heldout_dataset_{ds}",
        [d for d in joined if d.base.dataset != ds],
        [d for d in joined if d.base.dataset == ds],
    ) for ds in DATASETS]
    return {
        "schema": "awn_g5a_exploratory_evidence_ablation_v1",
        "source_g2_sha": SOURCE_GRID_SHA, "source_g2_run": SOURCE_G2_RUN,
        "feature_replay_commit": replay_sha,
        "historical_checkpoint_count": len(joined),
        "baseline_feature_names": list(BASE_NAMES),
        "new_feature_names": list(PROPOSAL_NAMES),
        "fixed_alpha": RIDGE_ALPHA, "new_future_outcome_rollouts": 0,
        "test_examples_used": 0,
        "seed_holdout": {"folds": seed_folds, "summary": summarize(seed_folds)},
        "dataset_holdout": {"folds": dataset_folds, "summary": summarize(dataset_folds)},
        "augmented_full_history_candidate_not_independently_validated": fit_augmented(joined),
        "interpretation": (
            "Exploratory historical G2 feature ablation only; G4 outcomes not used. "
            "Unseen-seed prospective precommitted confirmation remains necessary."
        ), "passed": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--g2-root", type=Path, required=True)
    p.add_argument("--features-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = evaluate(load_reports(args.g2_root), args.features_root)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite ablation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"], "source_g2_sha": result["source_g2_sha"],
        "seed_holdout": result["seed_holdout"]["summary"],
        "dataset_holdout": result["dataset_holdout"]["summary"],
        "new_rollouts": 0, "test_examples_used": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
