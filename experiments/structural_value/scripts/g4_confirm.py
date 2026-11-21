"""Stage B: execute matched native-AWN forks only after a saved prediction artifact.

The decision rule and every selected action must be read from the immutable
Stage-A output. No treatment outcome is available before all decisions
and snapshot hashes have been validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from ..actions.action_spec import ActionKind, ActionSpec
from ..adapters.awn_adapter import AWNAdapter
from ..counterfactual.fork_runner import paired_rollouts
from ..counterfactual.label_builder import build_labels
from ..g4_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, HORIZON_STEPS, ACTIONS,
    COST_WEIGHT, REFERENCE_SECONDS, G3_SOURCE_RUN, G3_SOURCE_ARTIFACT_ID,
    G4_ORIGINAL_PRECOMMIT_SHA,
)
from ..g4_frozen_policy import FrozenPolicy, _json_sha, verify_precommitted_prediction
from ..state.snapshot import TrainingSnapshot
from .audit_pilot import audit_pilot
from .pilot import _atomic_jsonl, _git_sha, future_batches
from .run_grid import _digest
from .value_predictor import ValueModel
from .g4_precommit import file_sha


def _json_digest(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def verify_precommit(pre: dict, input_dir: Path, *,
                     current_git_sha: str) -> FrozenPolicy:
    if pre.get("schema") != "awn_structural_value_g4_precommitted_predictions_v1":
        raise AssertionError("G4 prediction artifact is missing or unapproved")
    design = pre["design"]
    if (design["dataset"] not in DATASETS or design["seed"] not in SEEDS
            or design["checkpoint_epochs"] != list(CHECKPOINTS)
            or design["rollout_horizons"] != list(HORIZONS)
            or design["horizon_steps"] != HORIZON_STEPS
            or design["actions"] != list(ACTIONS)
            or design["cost_weight"] != COST_WEIGHT
            or design["reference_seconds"] != REFERENCE_SECONDS
            or design["source_g3_run"] != G3_SOURCE_RUN
            or design["source_g3_artifact_id"] != G3_SOURCE_ARTIFACT_ID
            or design["source_commit"] != G4_ORIGINAL_PRECOMMIT_SHA
            or len(current_git_sha) != 40
            or design["test_examples_for_labels"] != 0
            or design["learning_rate"] != 0.01
            or design["device"] != "cpu"):
        raise AssertionError("Changed preregistered prospective experiment")
    if (pre["design_hash"] != _json_digest(design)
            or pre["test_examples_used"] != 0
            or pre["split"]["test_examples_used"] != 0
            or pre["outcome_files_created"] != 0
            or pre["status"] != "predictions_precommitted_before_interventions"
            or len(pre["checkpoint_decisions"]) != len(CHECKPOINTS)):
        raise AssertionError("Precommit integrity or completeness invalid")
    model = pre["locked_model"]
    if _json_sha(model) != design["frozen_model_sha256"]:
        raise AssertionError("G3 model altered between planning and intervention")
    if (model["alpha_fixed"] != 12.0
            or model["target_actions"] != [
                "random_positional", "weight_positional", "gradient_positional"
            ] or model["feature_names"] != [
                "epoch", "physical_width", "learned_width", "width_rate",
                "train_elbo_loss", "validation_ce", "validation_accuracy",
                "validation_ce_change_last_five_epochs", "training_steps_so_far",
                "is_spiral_hard"
            ]):
        raise AssertionError("Frozen Ridge method changed")
    policy = FrozenPolicy(
        ValueModel(
            np.array(model["mean"], dtype=np.float64),
            np.array(model["std"], dtype=np.float64),
            np.array(model["beta"], dtype=np.float64)),
        design["frozen_model_sha256"], G3_SOURCE_RUN)
    history = pre["baseline_training_history"]
    if (len(history) != CHECKPOINTS[-1]
            or [h["epoch"] for h in history] != list(range(1, CHECKPOINTS[-1] + 1))):
        raise AssertionError("Baseline AWN training trajectory incomplete")
    if [p["epoch"] for p in pre["checkpoint_decisions"]] != list(CHECKPOINTS):
        raise AssertionError("Prospective decisions reordered or missing")
    if len({p["checkpoint_id"] for p in pre["checkpoint_decisions"]}) != 4:
        raise AssertionError("Duplicated AWN checkpoint identity")
    for d in pre["checkpoint_decisions"]:
        epoch = d["epoch"]
        if d["model_sha256"] != policy.model_sha256:
            raise AssertionError("Inconsistent frozen model")
        expected_seed = 100000 + design["seed"] * 13 + epoch * 997
        if d["rollout_seed"] != expected_seed:
            raise AssertionError("New seed's post-checkpoint future stream changed")
        if not d["checkpoint_id"].startswith(
                f"{pre['run_id']}:epoch{epoch:04d}:step"):
            raise AssertionError("Checkpoint provenance invalid")
        state = d["state"]
        if (state["epoch"] != epoch
                or state["physical_width"] != history[epoch - 1]["physical_width"]
                or state["learned_width"] != history[epoch - 1]["learned_width"]
                or state["validation_ce"] != history[epoch - 1]["validation_ce"]
                or state["train_elbo_loss"] != history[epoch - 1]["mean_train_elbo_loss"]
                or state["training_steps_so_far"] != sum(
                    h["training_steps"] for h in history[:epoch])):
            raise AssertionError("Prediction state did not come from prior baseline")
        verify_precommitted_prediction(
            policy, state=state, dataset=design["dataset"],
            recorded_action=d["predicted_action"],
            recorded_values=d["predicted_net_value"])
        filename = f"epoch{epoch:04d}.pt"
        path = input_dir / filename
        if d["snapshot_file"] != filename or not path.is_file():
            raise AssertionError("Original preregistered checkpoint missing")
        if file_sha(path) != d["snapshot_sha256"]:
            raise AssertionError("Model/optimizer snapshot changed after prediction")
    return policy


def confirm(*, source: Path, output: Path) -> dict:
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite confirmation outcomes")
    precommit_path = source / "precommit.json"
    pre = json.loads(precommit_path.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    git_sha = _git_sha()
    policy = verify_precommit(pre, source, current_git_sha=git_sha)
    # All predicted actions are validated before the first intervention.
    from experiments.bottleneck_triggered.common_utils import load_dataset_splits
    from experiments.bottleneck_triggered.evidence_sweep.train_screen import safe_loaders
    ds, seed = pre["design"]["dataset"], pre["design"]["seed"]
    original_train, original_val, original_test, _ = load_dataset_splits(ds)
    train, calibration, validation, split = safe_loaders(
        original_train, original_val, original_test, seed=seed)
    if split != pre["split"] or len(train.dataset) != pre["train_examples"]:
        raise AssertionError("Training/validation/calibration split changed")
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    for d in pre["checkpoint_decisions"]:
        epoch, checkpoint_id = d["epoch"], d["checkpoint_id"]
        # Restricted to locally generated, hashed GitHub Action artifacts.
        # weights_only=False is required for the genuine AWN object graph.
        snap = torch.load(source / d["snapshot_file"],
                          weights_only=False, map_location="cpu")
        if not isinstance(snap, TrainingSnapshot):
            raise AssertionError("Saved object is not a native AWN checkpoint")
        prov = snap.state.provenance
        if (prov["dataset"] != ds or prov["seed"] != seed
                or prov["checkpoint_id"] != checkpoint_id
                or prov["split"] != split):
            raise AssertionError("Snapshot identity does not match precommit")
        physical = int(snap.state.model.dyn_model.get_layer(0).weight.shape[0])
        if physical != d["state"]["physical_width"]:
            raise AssertionError("Snapshot physical AWN width differs")
        if (int(snap.state.model.variational_Ws[0]
                .compute_truncation_number()) != d["state"]["learned_width"]):
            raise AssertionError("Snapshot native learned AWN width differs")
        rate = float(snap.state.model.variational_Ws[0]
                     .discretized_distribution.base_distribution.rate
                     .detach().cpu().item())
        if abs(rate - d["state"]["width_rate"]) > 1e-10:
            raise AssertionError("Snapshot width rate differs")
        batches = future_batches(train, seed=d["rollout_seed"],
                                 epochs=1, max_steps=HORIZON_STEPS)
        if len(batches) != HORIZON_STEPS:
            raise AssertionError("Prospective matched future batch stream truncated")
        actions = [
            ActionSpec(ActionKind.SKIP),
            ActionSpec(ActionKind.RANDOM_POS, action_seed=d["rollout_seed"] + 301),
            ActionSpec(ActionKind.WEIGHT_POS),
            ActionSpec(ActionKind.GRADIENT_POS),
        ]

        def factory(restored):
            return AWNAdapter(
                restored, calibration=calibration, validation=validation,
                train_examples=pre["train_examples"], device=torch.device("cpu"),
                namespace=checkpoint_id, epoch=epoch)

        rows = paired_rollouts(
            snap, factory, batches, actions, HORIZONS,
            run_id=pre["run_id"], checkpoint_id=checkpoint_id,
            rollout_seed=d["rollout_seed"])
        labels = build_labels(rows, cost_weight=COST_WEIGHT,
                              reference_seconds=REFERENCE_SECONDS)
        config = {
            **pre["design"], "checkpoint_epoch": epoch,
            "rollout_seed": d["rollout_seed"],
            "horizon_epochs": 1, "max_rollout_steps": HORIZON_STEPS,
            "coupled_control": False, "git_sha": git_sha,
        }
        pilot_manifest = {
            "schema": "structural_value_pilot_v1",
            "config": config, "config_hash": _json_digest(config),
            "run_id": pre["run_id"], "checkpoint_id": checkpoint_id,
            "split": split, "training_population": pre["train_examples"],
            "rollout_optimizer_steps": len(batches),
            "rollout_endpoints": list(HORIZONS),
            "materialized_batch_sha256": _digest(batches),
            "actions": [a.payload for a in actions],
            "paired_observations": len(rows), "paired_labels": len(labels),
            "test_examples_used_for_selection": 0,
            "decision_state": d["state"],
            "precommitted_action": d["predicted_action"],
            "precommit_model_sha256": policy.model_sha256,
            "interpretation": "G4 true new-seed offline matched outcome confirmation",
        }
        cp_out = output / f"epoch{epoch:04d}"
        cp_out.mkdir()
        _atomic_jsonl(cp_out / "observations.jsonl", [r.as_dict() for r in rows])
        _atomic_jsonl(cp_out / "labels.jsonl", [r.as_dict() for r in labels])
        (cp_out / "manifest.json").write_text(json.dumps(
            pilot_manifest, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        audit = audit_pilot(cp_out)
        (cp_out / "audit.json").write_text(json.dumps(
            audit, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        h8 = audit["final_horizon_comparison"]
        by_action = {r["action"]: r for r in h8}
        if set(by_action) != set(ACTIONS):
            raise AssertionError("Incomplete action grid")
        chosen = d["predicted_action"]
        realized = by_action[chosen]
        checkpoints.append({
            "epoch": epoch, "checkpoint_id": checkpoint_id,
            "precommitted_action": chosen,
            "precommitted_net_value_predictions": d["predicted_net_value"],
            "precommitted_state": d["state"],
            "selected_observed_net_value": realized["net_value"],
            "selected_observed_delta_risk": realized["delta_risk"],
            "selected_no_op": realized["no_op"],
            "all_observed_actions_at_h8": h8,
            "audit_passed": audit["passed"], "intermediate_observations": 12,
            "snapshot_sha256": d["snapshot_sha256"],
        })
    shutil.copyfile(precommit_path, output / "precommit_original.json")
    result = {
        "schema": "awn_structural_value_g4_confirmation_v1",
        "source_commit": pre["design"]["source_commit"],
        "confirmation_engine_sha": git_sha,
        "source_precommit_sha256": file_sha(precommit_path),
        "source_g3_run": G3_SOURCE_RUN,
        "source_g3_artifact_id": G3_SOURCE_ARTIFACT_ID,
        "frozen_model_sha256": policy.model_sha256,
        "dataset": ds, "seed": seed, "run_id": pre["run_id"],
        "split": split, "checkpoint_results": checkpoints,
        "final_horizon_rollouts": len(checkpoints) * len(ACTIONS),
        "all_horizon_observations": len(checkpoints) * len(ACTIONS) * len(HORIZONS),
        "test_examples_used": 0, "passed": len(checkpoints) == len(CHECKPOINTS),
        "interpretation": "New-seed preregistered offline AWN counterfactual confirmation",
    }
    (output / "confirmation.json").write_text(json.dumps(
        result, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "dataset": ds, "seed": seed,
        "passed": result["passed"],
        "checkpoints": len(checkpoints),
        "rollouts": result["final_horizon_rollouts"],
        "selected_actions": [r["precommitted_action"] for r in checkpoints],
        "selected_realized_net_values": [
            r["selected_observed_net_value"] for r in checkpoints],
    }, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    confirm(**vars(args))


if __name__ == "__main__":
    main()
