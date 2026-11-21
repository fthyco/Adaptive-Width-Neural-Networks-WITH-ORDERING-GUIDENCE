"""Prospective action-value predictor with group-held-out offline evaluation.

All features are available before the intervention. Complete counterfactual
labels are used only for training (in training groups) or retrospective
evaluation (in held-out groups); never as prediction-time features.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ..grid_contract import ACTIONS, CHECKPOINTS, DATASETS, SEEDS
from .aggregate_grid import aggregate_reports
from .audit_grid import audit_grid

# Frozen source of the already completed G2 engineering grid.
SOURCE_GRID_SHA = "ffb347214a86e4fbf92912ff2bf4dac54039edd5"
SOURCE_GRID_RUN = 44848340275
TARGET_ACTIONS = tuple(a for a in ACTIONS if a != "skip")
RIDGE_ALPHA = 12.0  # Fixed before looking at held-out results. Never tuned on folds.
FEATURE_NAMES = (
    "epoch", "physical_width", "learned_width", "width_rate",
    "train_elbo_loss", "validation_ce", "validation_accuracy",
    "validation_ce_change_last_five_epochs", "training_steps_so_far",
    "is_spiral_hard",
)
STATE_KEYS = (
    "epoch", "physical_width", "learned_width", "width_rate",
    "train_elbo_loss", "validation_ce", "validation_accuracy",
    "validation_ce_change_last_five_epochs", "training_steps_so_far",
)


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError(f"Missing numeric predecision value: {name}")
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError(f"Nonfinite predecision value: {name}")
    return number


@dataclass(frozen=True)
class Decision:
    dataset: str
    seed: int
    epoch: int
    checkpoint_id: str
    x: tuple[float, ...]
    outcomes: dict[str, dict]

    @property
    def trajectory(self) -> tuple[str, int]:
        return (self.dataset, self.seed)


def load_reports(root: Path) -> list[dict]:
    """Re-audit original raw records, not only downloaded summary JSON."""
    results = []
    for dataset in DATASETS:
        for seed in SEEDS:
            directory = Path(root) / f"grid_{dataset}_seed{seed}"
            original_path = directory / "grid_audit.json"
            if not original_path.is_file():
                raise FileNotFoundError(f"Missing source grid audit: {original_path}")
            previous = json.loads(original_path.read_text(encoding="utf-8"))
            fresh = audit_grid(directory)
            if fresh != previous:
                raise AssertionError("Raw fork records disagree with stored source audit")
            if fresh["git_sha"] != SOURCE_GRID_SHA:
                raise AssertionError("Incorrect 96-rollout source commit")
            results.append(fresh)
    return results


def build_decisions(reports: Sequence[dict]) -> list[Decision]:
    """Construct one factual predecision feature vector per AWN checkpoint."""
    complete = aggregate_reports(list(reports))
    if not complete["passed"] or complete["git_sha"] != SOURCE_GRID_SHA:
        raise AssertionError("Unapproved, incomplete or mixed source grid")
    decisions: list[Decision] = []
    seen = set()
    for report in reports:
        for cp in report["checkpoints"]:
            state = cp["decision_state"]
            if set(state) != set(STATE_KEYS):
                raise AssertionError("Unapproved feature / missing predecision state")
            epoch = cp["epoch"]
            if state["epoch"] != epoch:
                raise AssertionError("Decision epoch and source features disagree")
            if epoch not in CHECKPOINTS:
                raise AssertionError("Unpredeclared checkpoint")
            # Training / held-out labels are intentionally not accessed here.
            features = tuple(_finite(state[n], n) for n in STATE_KEYS) + (
                float(report["dataset"] == "SpiralHard"),
            )
            rows = cp["final_horizon_comparison"]
            if len(rows) != len(ACTIONS) or len({r["action"] for r in rows}) != len(ACTIONS):
                raise AssertionError("Non-bijective checkpoint action labels")
            outcomes = {}
            skip_risk = next(r["validation_ce"] for r in rows
                             if r["action"] == "skip")
            for row in rows:
                action = row["action"]
                if action not in ACTIONS or row["gate_mode"] != "positional":
                    raise AssertionError("Unsupported action semantics")
                for key in ("net_value", "delta_risk", "validation_ce",
                            "delta_online_seconds"):
                    _finite(row[key], key)
                if abs((skip_risk - row["validation_ce"]) - row["delta_risk"]) > 1e-8:
                    raise AssertionError("Action has invalid paired risk label")
                if action == "skip" and (
                    row["net_value"] != 0 or row["delta_risk"] != 0 or not row["no_op"]
                ):
                    raise AssertionError("Missing zero-valued untreated reference")
                outcomes[action] = {
                    "net_value": float(row["net_value"]),
                    "delta_risk": float(row["delta_risk"]),
                    "no_op": bool(row["no_op"]),
                    "delta_online_seconds": float(row["delta_online_seconds"]),
                }
            if set(outcomes) != set(ACTIONS):
                raise AssertionError("Missing action label")
            key = (report["dataset"], report["seed"], epoch)
            if key in seen or not cp["checkpoint_id"]:
                raise AssertionError("Duplicate checkpoint decision")
            seen.add(key)
            decisions.append(Decision(
                report["dataset"], report["seed"], epoch,
                cp["checkpoint_id"], features, outcomes))
    if len(decisions) != 24:
        raise AssertionError("Must have 24 original decision contexts")
    if len({d.checkpoint_id for d in decisions}) != 24:
        raise AssertionError("Nonunique checkpoint identities")
    return sorted(decisions, key=lambda d: (d.dataset, d.seed, d.epoch))


@dataclass(frozen=True)
class ValueModel:
    """Train-only standardized, fixed-L2 multi-output ridge with unpenalized intercept."""
    mean: np.ndarray
    std: np.ndarray
    beta: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def predict(self, decisions: Sequence[Decision]) -> np.ndarray:
        x = np.asarray([d.x for d in decisions], dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError("Missing or unapproved decision-time features")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            z = (x - self.mean) / self.std
            pred = np.column_stack((np.ones(len(x)), z)) @ self.beta
        if not np.isfinite(pred).all():
            raise AssertionError("Nonfinite predicted intervention value")
        return pred

    def as_dict(self) -> dict:
        return {
            "type": "train_only_standardized_multioutput_ridge",
            "alpha_fixed": RIDGE_ALPHA,
            "feature_names": list(self.feature_names),
            "target_actions": list(TARGET_ACTIONS),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "beta": self.beta.tolist(),
        }


def fit_value_model(training: Sequence[Decision]) -> ValueModel:
    if len(training) < 8 or len({d.checkpoint_id for d in training}) != len(training):
        raise ValueError("Too few or duplicated training checkpoint decisions")
    x = np.array([d.x for d in training], dtype=np.float64)
    y = np.array([[d.outcomes[a]["net_value"] for a in TARGET_ACTIONS]
                  for d in training], dtype=np.float64)
    if x.shape[1] != len(FEATURE_NAMES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError("Invalid train features or labels")
    # No standardized statistics or targets come from the held-out groups.
    mean = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    z = (x - mean) / sd
    design = np.column_stack((np.ones(len(training)), z))
    reg = np.eye(design.shape[1], dtype=np.float64) * RIDGE_ALPHA
    reg[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return ValueModel(mean, sd, beta)


def decide(prediction: Sequence[float]) -> str:
    values = [0.0] + [float(v) for v in prediction]
    if len(values) != len(ACTIONS) or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid predicted action values")
    # Predeclared abstention threshold = 0; ties prefer unchanged AWN.
    return ACTIONS[int(np.argmax(values))]


def _static_train_best(training: Sequence[Decision]) -> str:
    scores = [0.0] + [
        float(np.mean([d.outcomes[a]["net_value"] for d in training]))
        for a in TARGET_ACTIONS
    ]
    return ACTIONS[int(np.argmax(scores))]


def _one_fold(name: str, training: Sequence[Decision],
              heldout: Sequence[Decision]) -> dict:
    train = list(training)
    test = list(heldout)
    if not train or not test:
        raise AssertionError("Empty fold")
    if set(d.trajectory for d in train) & set(d.trajectory for d in test):
        raise AssertionError("Training/checkpoint trajectory leaked into evaluation")
    if set(d.checkpoint_id for d in train) & set(d.checkpoint_id for d in test):
        raise AssertionError("Identical checkpoint used in fit and evaluation")
    model = fit_value_model(train)
    preds = model.predict(test)
    static_action = _static_train_best(train)
    rows = []
    for d, estimated in zip(test, preds):
        policy = decide(estimated)
        oracle = max(ACTIONS, key=lambda a: d.outcomes[a]["net_value"])
        values = {
            "ridge_policy": d.outcomes[policy]["net_value"],
            "train_mean_static": d.outcomes[static_action]["net_value"],
            "always_skip": 0.0,
            "always_random": d.outcomes["random_positional"]["net_value"],
            "always_w": d.outcomes["weight_positional"]["net_value"],
            "always_g": d.outcomes["gradient_positional"]["net_value"],
            # Held-out oracle is diagnostic and **never** a deployable policy.
            "hindsight_oracle_upper_bound": d.outcomes[oracle]["net_value"],
        }
        prediction_error = {
            action: abs(float(estimated[j]) - d.outcomes[action]["net_value"])
            for j, action in enumerate(TARGET_ACTIONS)
        }
        rows.append({
            "dataset": d.dataset, "seed": d.seed, "epoch": d.epoch,
            "checkpoint_id": d.checkpoint_id,
            "predicted_net_value": {
                a: float(estimated[i]) for i, a in enumerate(TARGET_ACTIONS)
            },
            "predicted_action": policy, "train_static_action": static_action,
            "observed_selected_net_value": values["ridge_policy"],
            "observed_selected_delta_risk": d.outcomes[policy]["delta_risk"],
            "observed_action_values": {
                a: d.outcomes[a]["net_value"] for a in ACTIONS
            },
            "observed_policy_values": values,
            "absolute_error": prediction_error,
            "hindsight_best_action_diagnostic": oracle,
            "selected_action_no_op": d.outcomes[policy]["no_op"],
        })
    return {
        "fold": name,
        "train_trajectories": sorted([list(x) for x in {
            d.trajectory for d in train
        }]),
        "heldout_trajectories": sorted([list(x) for x in {
            d.trajectory for d in test
        }]),
        "train_checkpoint_count": len(train),
        "heldout_checkpoint_count": len(test),
        "fitted_model": model.as_dict(),
        "train_static_action": static_action,
        "decisions": rows,
    }


def _summary(folds: list[dict]) -> dict:
    rows = [row for fold in folds for row in fold["decisions"]]
    if len(rows) != 24 or len({x["checkpoint_id"] for x in rows}) != 24:
        raise AssertionError("Held-out coverage must be exactly 24 decisions")
    policies = (
        "ridge_policy", "train_mean_static", "always_skip",
        "always_random", "always_w", "always_g",
        "hindsight_oracle_upper_bound",
    )
    return {
        "heldout_decisions": len(rows),
        "policies": {
            name: {
                "mean_net_value": float(np.mean([
                    r["observed_policy_values"][name] for r in rows
                ])),
                "median_net_value": float(np.median([
                    r["observed_policy_values"][name] for r in rows
                ])),
                "beneficial_decisions": sum(
                    r["observed_policy_values"][name] > 0 for r in rows
                ),
                "harmful_decisions": sum(
                    r["observed_policy_values"][name] < 0 for r in rows
                ),
                "min_net_value": float(min(
                    r["observed_policy_values"][name] for r in rows
                )),
            } for name in policies
        },
        "ridge_action_counts": {
            a: sum(r["predicted_action"] == a for r in rows) for a in ACTIONS
        },
        "mae_by_action": {
            a: float(np.mean([r["absolute_error"][a] for r in rows]))
            for a in TARGET_ACTIONS
        },
        "selected_noop_decisions": sum(
            r["selected_action_no_op"] for r in rows),
        "fold_mean_policy_net_value": {
            fold["fold"]: float(np.mean([
                r["observed_policy_values"]["ridge_policy"]
                for r in fold["decisions"]
            ])) for fold in folds
        },
    }


def evaluate_group_heldout(reports: Sequence[dict]) -> dict:
    """One fixed model across predeclared outer folds, no hyperparameter search."""
    decisions = build_decisions(reports)
    # Both tasks for a seed are held out together; checkpoints of a
    # trajectory are never split between train and evaluation.
    seed_folds = [
        _one_fold(
            f"heldout_seed_{seed}",
            [d for d in decisions if d.seed != seed],
            [d for d in decisions if d.seed == seed],
        ) for seed in SEEDS
    ]
    # A separate, more difficult leave-dataset-out diagnostic. Do not
    # pool these 24 decisions with the seed-heldout outcomes as if
    # there were 48 independent experimental decision units.
    dataset_folds = [
        _one_fold(
            f"heldout_dataset_{ds}",
            [d for d in decisions if d.dataset != ds],
            [d for d in decisions if d.dataset == ds],
        ) for ds in DATASETS
    ]
    return {
        "schema": "awn_structural_value_predictor_groupheldout_v1",
        "source_grid_sha": SOURCE_GRID_SHA,
        "source_grid_run": SOURCE_GRID_RUN,
        "label": "net_value = delta_risk - 0.01 * incremental_online_seconds",
        "label_origin": "fully observed, paired H8 grid96 fork labels",
        "predictor": "standardized_multioutput_ridge",
        "fixed_alpha": RIDGE_ALPHA,
        "zero_abstention_threshold": True,
        "feature_names": list(FEATURE_NAMES),
        "target_actions": list(TARGET_ACTIONS),
        "training_decision_count": len(decisions),
        "fold_protocols": {
            "leave_one_seed_out": {
                "split_definition": "withhold both dataset trajectories for one seed",
                "folds": seed_folds, "summary": _summary(seed_folds),
            },
            "leave_one_dataset_out": {
                "split_definition": "withhold all three seed trajectories of one dataset",
                "folds": dataset_folds, "summary": _summary(dataset_folds),
            },
        },
        # Candidate is trained on all 24 after cross-validated evaluation.
        # Its in-sample performance is neither computed nor advertised.
        "full_data_candidate_not_independently_validated": (
            fit_value_model(decisions).as_dict()
        ),
        "test_examples_used": 0,
        "passed": True,
        "interpretation": (
            "Exploratory offline held-out labels, not deployment, statistical "
            "significance, or generalization to untouched test data"
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_group_heldout(load_reports(args.root))
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite previously evaluated model")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overview = {
        "passed": result["passed"],
        "source_grid_run": result["source_grid_run"],
        "n_decisions": result["training_decision_count"],
        "seed_holdout": result["fold_protocols"]["leave_one_seed_out"]["summary"],
        "dataset_holdout": result["fold_protocols"]["leave_one_dataset_out"]["summary"],
    }
    print(json.dumps(overview, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
