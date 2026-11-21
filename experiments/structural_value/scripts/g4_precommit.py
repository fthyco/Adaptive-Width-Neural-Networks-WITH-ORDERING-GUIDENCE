"""Stage A: persist locked predictions before any new-seed counterfactual label exists.

This executable can only train native AWN baselines, evaluate predecision
development CE and save checkpoints and frozen G3 policy decisions.
It does NOT fork, intervene, compute future labels or read final test labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..g4_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, HORIZON_STEPS, ACTIONS,
    COST_WEIGHT, REFERENCE_SECONDS, G3_SOURCE_ARTIFACT_ID,
)
from ..g4_frozen_policy import load_frozen_policy
from ..state.snapshot import TrainingSnapshot
from .pilot import _git_sha


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(*, dataset: str, seed: int, g3_artifact: Path, output: Path) -> dict:
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("Prospective dataset or seed not predeclared")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to mix or overwrite prospective predictions")
    policy = load_frozen_policy(g3_artifact)
    from experiments.bottleneck_triggered.common_utils import (
        load_dataset_splits, get_awn_config)
    from experiments.bottleneck_triggered.evidence_sweep.train_screen import (
        safe_loaders, train_epoch, check_validation)
    from metric import ELBO_Classification
    from model import AWN

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw_train, raw_val, raw_test, _ = load_dataset_splits(dataset)
    train, calibration, validation, split = safe_loaders(
        raw_train, raw_val, raw_test, seed=seed)
    if split["test_examples_used"] != 0:
        raise AssertionError("Confirmation cannot use final test labels")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    n_train = len(train.dataset)
    model = AWN(2, 2, get_awn_config(n_train, enable_learned_ordering=False))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = optimizer
    criterion = ELBO_Classification(n_train)
    sha = _git_sha()
    design = {
        "dataset": dataset, "seed": seed,
        "checkpoint_epochs": list(CHECKPOINTS),
        "rollout_horizons": list(HORIZONS),
        "horizon_steps": HORIZON_STEPS,
        "actions": list(ACTIONS),
        "cost_weight": COST_WEIGHT,
        "reference_seconds": REFERENCE_SECONDS,
        "source_g3_artifact_id": G3_SOURCE_ARTIFACT_ID,
        "source_g3_run": policy.artifact_run_id,
        "frozen_model_sha256": policy.model_sha256,
        "source_commit": sha, "device": "cpu",
        "learning_rate": 0.01,
        "reference_data": "original_repository_dataset_and_split",
        "test_examples_for_labels": 0,
    }
    design_hash = hashlib.sha256(json.dumps(
        design, sort_keys=True).encode()).hexdigest()
    run_id = f"{dataset}:seed{seed}:prospective:{design_hash[:16]}:{sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    history: list[dict] = []
    decisions = []
    steps = 0
    for epoch in range(1, CHECKPOINTS[-1] + 1):
        record = train_epoch(model, train, optimizer, criterion, device)
        record.update(check_validation(model, validation, device))
        record["epoch"] = epoch
        record["physical_width"] = int(
            model.dyn_model.get_layer(0).weight.shape[0])
        record["learned_width"] = int(
            model.variational_Ws[0].compute_truncation_number())
        history.append(record)
        steps += record["training_steps"]
        if epoch not in CHECKPOINTS:
            continue

        checkpoint_id = f"{run_id}:epoch{epoch:04d}:step{steps}"
        rollout_seed = 100000 + seed * 13 + epoch * 997
        previous = history[max(0, epoch - 6)]
        state = {
            "epoch": epoch,
            "physical_width": record["physical_width"],
            "learned_width": record["learned_width"],
            "width_rate": float(
                model.variational_Ws[0]
                    .discretized_distribution.base_distribution.rate
                    .detach().cpu().item()),
            "train_elbo_loss": record["mean_train_elbo_loss"],
            "validation_ce": record["validation_ce"],
            "validation_accuracy": record["validation_accuracy"],
            "validation_ce_change_last_five_epochs": (
                previous["validation_ce"] - record["validation_ce"]),
            "training_steps_so_far": steps,
        }
        action, predictions = policy.action(state, dataset=dataset)
        snapshot = TrainingSnapshot.capture(
            model, optimizer, loader=train, history=history,
            provenance={"dataset": dataset, "seed": seed,
                        "checkpoint_id": checkpoint_id, "split": split})
        cp_name = f"epoch{epoch:04d}.pt"
        # Snapshot is generated exclusively inside the trusted repository
        # runner, never accepted from an arbitrary external input.
        torch.save(snapshot, output / cp_name)
        decisions.append({
            "epoch": epoch, "checkpoint_id": checkpoint_id,
            "rollout_seed": rollout_seed, "state": state,
            "predicted_action": action,
            "predicted_net_value": predictions,
            "snapshot_file": cp_name,
            "snapshot_sha256": file_sha(output / cp_name),
            "model_sha256": policy.model_sha256,
        })
    if len(decisions) != len(CHECKPOINTS):
        raise AssertionError("Prospective prediction set incomplete")
    precommit = {
        "schema": "awn_structural_value_g4_precommitted_predictions_v1",
        "design": design, "design_hash": design_hash,
        "run_id": run_id, "split": split, "train_examples": n_train,
        "checkpoint_decisions": decisions,
        "baseline_training_history": history,
        "locked_model": policy.model.as_dict(),
        "test_examples_used": 0,
        "outcome_files_created": 0,
        "status": "predictions_precommitted_before_interventions",
    }
    (output / "precommit.json").write_text(json.dumps(
        precommit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "dataset": dataset, "seed": seed,
        "predicted_checkpoint_decisions": len(decisions),
        "decisions": [
            {"epoch": d["epoch"], "action": d["predicted_action"],
             "predictions": d["predicted_net_value"]}
            for d in decisions
        ],
        "frozen_model_sha256": policy.model_sha256,
        "test_examples_used": 0,
        "intervention_rollouts_executed": 0,
    }, indent=2))
    return precommit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--g3-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(**vars(args))


if __name__ == "__main__":
    main()
