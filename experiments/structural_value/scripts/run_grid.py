"""Run exactly four fixed-epoch AWN checkpoints per reference task and seed.

Each (dataset, seed) trains one uninterrupted native AWN baseline, with
independent checkpoint forks only for offline counterfactual teacher labels.
A fixed future batch stream is shared by treatments within each checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..actions.action_spec import ActionKind, ActionSpec
from ..adapters.awn_adapter import AWNAdapter
from ..counterfactual.fork_runner import paired_rollouts
from ..counterfactual.label_builder import build_labels
from ..state.snapshot import TrainingSnapshot
from .audit_pilot import audit_pilot
from .pilot import _atomic_jsonl, _git_sha, future_batches
from ..grid_contract import DATASETS, SEEDS, CHECKPOINTS, HORIZON_STEPS, HORIZONS


def _digest(batches):
    h = hashlib.sha256()
    for x, y in batches:
        h.update(x.contiguous().numpy().tobytes())
        h.update(y.contiguous().numpy().tobytes())
    return h.hexdigest()


def run_grid(*, dataset: str, seed: int, output: Path,
             checkpoint_epochs: tuple[int, ...] = CHECKPOINTS,
             max_rollout_steps: int = HORIZON_STEPS,
             cost_weight: float = 0.01, reference_seconds: float = 1.0) -> dict:
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("Undeclared reference dataset or research seed")
    if (tuple(checkpoint_epochs) != CHECKPOINTS
            or max_rollout_steps != HORIZON_STEPS):
        raise ValueError("Cannot change the predeclared 96-rollout design")
    if cost_weight != 0.01 or reference_seconds != 1.0:
        raise ValueError("Cannot post-hoc tune the declared cost exchange rate")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Grid output is immutable; refusing overwrite")
    from experiments.bottleneck_triggered.common_utils import (
        get_awn_config, load_dataset_splits)
    from experiments.bottleneck_triggered.evidence_sweep.train_screen import (
        safe_loaders, train_epoch, check_validation)
    from metric import ELBO_Classification
    from model import AWN

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    # The reference dataset factory may set NumPy's global seed. Re-seed
    # only after constructing all datasets and checked disjoint subsets.
    raw_train, raw_val, raw_test, _ = load_dataset_splits(dataset)
    train, calibration, validation, split = safe_loaders(
        raw_train, raw_val, raw_test, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    train_n = len(train.dataset)
    model = AWN(2, 2, get_awn_config(train_n, enable_learned_ordering=False))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = optimizer
    elbo = ELBO_Classification(train_n)
    git_sha = _git_sha()

    design = {
        "dataset": dataset, "seed": seed, "checkpoint_epochs": list(CHECKPOINTS),
        "max_rollout_steps": max_rollout_steps, "horizons": list(HORIZONS),
        "actions": ["skip", "random_positional", "weight_positional",
                    "gradient_positional"],
        "cost_weight": cost_weight, "reference_seconds": reference_seconds,
        "git_sha": git_sha, "device": "cpu", "learning_rate": 0.01,
        "reference_data": "original_repository_dataset_and_split",
        "calibration_validation": "development_only",
        "test_examples_for_labels": 0,
    }
    design_hash = hashlib.sha256(
        json.dumps(design, sort_keys=True).encode()).hexdigest()
    run_id = f"{dataset}:seed{seed}:grid96:{design_hash[:16]}:{git_sha[:12]}"

    output.mkdir(parents=True, exist_ok=False)
    history: list[dict] = []
    checkpoint_reports: list[dict] = []
    observed_steps = 0
    for epoch in range(1, max(CHECKPOINTS) + 1):
        epoch_result = train_epoch(model, train, optimizer, elbo, device)
        epoch_result.update(check_validation(model, validation, device))
        epoch_result["epoch"] = epoch
        epoch_result["physical_width"] = int(
            model.dyn_model.get_layer(0).weight.shape[0])
        epoch_result["learned_width"] = int(
            model.variational_Ws[0].compute_truncation_number())
        history.append(epoch_result)
        observed_steps += epoch_result["training_steps"]
        if epoch not in CHECKPOINTS:
            continue

        checkpoint_id = f"{run_id}:epoch{epoch:04d}:step{observed_steps}"
        rollout_seed = int(100000 + seed * 13 + epoch * 997)
        snapshot = TrainingSnapshot.capture(
            model, optimizer, loader=train, history=history,
            provenance={"dataset": dataset, "seed": seed,
                        "checkpoint_id": checkpoint_id, "split": split})
        batches = future_batches(
            train, seed=rollout_seed, epochs=1, max_steps=max_rollout_steps)
        if len(batches) != max_rollout_steps:
            raise RuntimeError("Reference training split cannot cover horizon")
        actions = [
            ActionSpec(ActionKind.SKIP),
            ActionSpec(ActionKind.RANDOM_POS, action_seed=rollout_seed + 301),
            ActionSpec(ActionKind.WEIGHT_POS),
            ActionSpec(ActionKind.GRADIENT_POS),
        ]

        def factory(restored):
            return AWNAdapter(
                restored, calibration=calibration, validation=validation,
                train_examples=train_n, device=device,
                namespace=checkpoint_id, epoch=epoch)

        try:
            observations = paired_rollouts(
                snapshot, factory, batches, actions, HORIZONS,
                run_id=run_id, checkpoint_id=checkpoint_id,
                rollout_seed=rollout_seed)
            labels = build_labels(
                observations, cost_weight=cost_weight,
                reference_seconds=reference_seconds)
        finally:
            # Forking changes global RNG state. The live AWN training stream
            # must continue from its original post-checkpoint RNG/sampler.
            snapshot.restore_rng(loader=train)

        config = {
            **design, "checkpoint_epoch": epoch, "rollout_seed": rollout_seed,
            "horizon_epochs": 1, "coupled_control": False,
            "checkpoint_set": list(CHECKPOINTS),
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()
        previous = history[max(0, epoch - 6)]
        decision_state = {
            "epoch": epoch,
            "physical_width": epoch_result["physical_width"],
            "learned_width": epoch_result["learned_width"],
            "width_rate": float(model.variational_Ws[0]
                                .discretized_distribution.base_distribution.rate
                                .detach().cpu().item()),
            "train_elbo_loss": epoch_result["mean_train_elbo_loss"],
            "validation_ce": epoch_result["validation_ce"],
            "validation_accuracy": epoch_result["validation_accuracy"],
            "validation_ce_change_last_five_epochs":
                (previous["validation_ce"] - epoch_result["validation_ce"]),
            "training_steps_so_far": observed_steps,
        }
        cp_dir = output / f"epoch{epoch:04d}"
        cp_dir.mkdir()
        _atomic_jsonl(
            cp_dir / "observations.jsonl", [r.as_dict() for r in observations])
        _atomic_jsonl(cp_dir / "labels.jsonl", [l.as_dict() for l in labels])
        cp_manifest = {
            "schema": "structural_value_pilot_v1",
            "config": config, "config_hash": config_hash,
            "run_id": run_id, "checkpoint_id": checkpoint_id,
            "split": split, "training_population": train_n,
            "rollout_optimizer_steps": len(batches),
            "rollout_endpoints": list(HORIZONS),
            "materialized_batch_sha256": _digest(batches),
            "actions": [a.payload for a in actions],
            "paired_observations": len(observations),
            "paired_labels": len(labels),
            "decision_state": decision_state,
            "test_examples_used_for_selection": 0,
            "interpretation": "96-rollout engineering study; not confirmatory",
        }
        (cp_dir / "manifest.json").write_text(
            json.dumps(cp_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        audit = audit_pilot(cp_dir)
        (cp_dir / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        checkpoint_reports.append({
            "epoch": epoch, "checkpoint_id": checkpoint_id,
            "decision_state": decision_state,
            "pilot_subdir": cp_dir.name,
            "observations": audit["observations"],
            "labels": audit["labels"],
            "audited": audit["passed"],
        })

    manifest = {
        "schema": "awn_structural_value_grid96_per_run_v1",
        "design": design, "design_hash": design_hash,
        "run_id": run_id, "git_sha": git_sha,
        "split": split, "training_population": train_n,
        "checkpoints": checkpoint_reports, "training_history": history,
        "total_final_horizon_rollouts": len(checkpoint_reports) * 4,
        "total_horizon_observations": len(checkpoint_reports) * 4 * len(HORIZONS),
        "test_examples_used_for_selection": 0,
        "passed": len(checkpoint_reports) == 4,
    }
    (output / "grid_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_grid(**vars(args))
    print(json.dumps({
        "dataset": result["design"]["dataset"],
        "seed": result["design"]["seed"], "passed": result["passed"],
        "checkpoints": [x["epoch"] for x in result["checkpoints"]],
        "final_rollouts": result["total_final_horizon_rollouts"],
        "all_observations": result["total_horizon_observations"],
    }, indent=2))


if __name__ == "__main__":
    main()
