"""One-run G5A: extract preaction evidence, then matched native AWN forks.

Unlike the failed replay attempt, features and labels are guaranteed to come
from the SAME captured state object; no historical checkpoint matching is
needed. This is exploratory grouped holdout, not prospective frozen-policy
confirmation. Gradient evidence is measured on calibration data only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from ..actions.action_spec import ActionKind, ActionSpec
from ..adapters.awn_adapter import AWNAdapter
from ..counterfactual.fork_runner import paired_rollouts
from ..counterfactual.label_builder import build_labels
from ..g5_fresh_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, HORIZON_STEPS, ACTIONS,
    COST_WEIGHT, REFERENCE_SECONDS,
)
from ..g5_evidence_features import FEATURE_NAMES, proposal_features
from ..state.snapshot import TrainingSnapshot
from .audit_pilot import audit_pilot
from .pilot import _atomic_jsonl, _git_sha, future_batches
from .run_grid import _digest


def run_study(*, dataset: str, seed: int, output: Path) -> dict:
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("Unregistered G5A cohort")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite G5A cohort")
    from experiments.bottleneck_triggered.common_utils import (
        get_awn_config, load_dataset_splits)
    from experiments.bottleneck_triggered.evidence_sweep.train_screen import (
        safe_loaders, train_epoch, check_validation)
    from experiments.bottleneck_triggered.evidence_sweep.controlled_bt import evidence_scores
    from metric import ELBO_Classification
    from model import AWN

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    raw_train, raw_val, raw_test, _ = load_dataset_splits(dataset)
    train, calibration, validation, split = safe_loaders(
        raw_train, raw_val, raw_test, seed=seed)
    if split["test_examples_used"] != 0:
        raise AssertionError("G5A cannot access final test labels")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    n_train = len(train.dataset)
    model = AWN(2, 2, get_awn_config(n_train, enable_learned_ordering=False))
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = optimizer
    criterion = ELBO_Classification(n_train)
    git_sha = _git_sha()
    design = {
        "dataset": dataset, "seed": seed, "checkpoint_epochs": list(CHECKPOINTS),
        "horizon_steps": HORIZON_STEPS, "horizons": list(HORIZONS),
        "actions": list(ACTIONS), "cost_weight": COST_WEIGHT,
        "reference_seconds": REFERENCE_SECONDS, "git_sha": git_sha,
        "device": "cpu", "learning_rate": 0.01,
        "label_source": "paired_forks_of_exact_predecision_feature_snapshot",
        "test_examples_used": 0,
    }
    design_hash = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    run_id = f"{dataset}:seed{seed}:g5a:{design_hash[:16]}:{git_sha[:12]}"
    output.mkdir(parents=True, exist_ok=False)
    history, checkpoints = [], []
    steps_so_far = 0
    for epoch in range(1, CHECKPOINTS[-1] + 1):
        entry = train_epoch(model, train, optimizer, criterion, dev)
        entry.update(check_validation(model, validation, dev))
        entry["epoch"] = epoch
        entry["physical_width"] = int(model.dyn_model.get_layer(0).weight.shape[0])
        entry["learned_width"] = int(model.variational_Ws[0].compute_truncation_number())
        history.append(entry)
        steps_so_far += entry["training_steps"]
        if epoch not in CHECKPOINTS:
            continue
        cp_id = f"{run_id}:epoch{epoch:04d}:step{steps_so_far}"
        rollout_seed = 100000 + seed * 13 + epoch * 997
        previous = history[max(0, epoch - 6)]
        state = {
            "epoch": epoch, "physical_width": entry["physical_width"],
            "learned_width": entry["learned_width"],
            "width_rate": float(model.variational_Ws[0]
                                .discretized_distribution.base_distribution.rate
                                .detach().cpu().item()),
            "train_elbo_loss": entry["mean_train_elbo_loss"],
            "validation_ce": entry["validation_ce"],
            "validation_accuracy": entry["validation_accuracy"],
            "validation_ce_change_last_five_epochs": (
                previous["validation_ce"] - entry["validation_ce"]),
            "training_steps_so_far": steps_so_far,
        }
        snapshot = TrainingSnapshot.capture(
            model, optimizer, loader=train, history=history,
            provenance={"dataset": dataset, "seed": seed, "checkpoint_id": cp_id, "split": split})
        try:
            started = perf_counter()
            shadow = snapshot.fork(loader=train).model
            ws = evidence_scores(shadow, calibration, "W", dev)
            gs = evidence_scores(shadow, calibration, "G", dev)
            features = proposal_features(
                w_scores=ws, g_scores=gs, learned_width=state["learned_width"],
                random_action_seed=rollout_seed + 301,
            )
            feature_seconds = perf_counter() - started
        finally:
            snapshot.restore_rng(loader=train)
        # Each checkpoint's preaction descriptor is complete BEFORE any
        # action is applied or future target is generated. Save a separate
        # preaction record for auditability.
        cp_dir = output / f"epoch{epoch:04d}"
        cp_dir.mkdir()
        preaction = {
            "checkpoint_id": cp_id, "dataset": dataset, "seed": seed,
            "epoch": epoch, "state": state, "feature_names": list(FEATURE_NAMES),
            "proposal_features": features,
            "feature_compute_seconds": feature_seconds,
            "future_outcomes_available": False,
        }
        (cp_dir / "preaction.json").write_text(
            json.dumps(preaction, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8")
        batches = future_batches(train, seed=rollout_seed, epochs=1, max_steps=HORIZON_STEPS)
        if len(batches) != HORIZON_STEPS:
            raise AssertionError("Not enough training-only future examples")
        actions = [
            ActionSpec(ActionKind.SKIP),
            ActionSpec(ActionKind.RANDOM_POS, action_seed=rollout_seed + 301),
            ActionSpec(ActionKind.WEIGHT_POS),
            ActionSpec(ActionKind.GRADIENT_POS),
        ]

        def adapter_factory(restored):
            return AWNAdapter(
                restored, calibration=calibration, validation=validation,
                train_examples=n_train, device=dev, namespace=cp_id, epoch=epoch)

        try:
            observations = paired_rollouts(
                snapshot, adapter_factory, batches, actions, HORIZONS,
                run_id=run_id, checkpoint_id=cp_id, rollout_seed=rollout_seed)
            labels = build_labels(
                observations, cost_weight=COST_WEIGHT,
                reference_seconds=REFERENCE_SECONDS)
        finally:
            snapshot.restore_rng(loader=train)
        _atomic_jsonl(cp_dir / "observations.jsonl", [x.as_dict() for x in observations])
        _atomic_jsonl(cp_dir / "labels.jsonl", [x.as_dict() for x in labels])
        config = {
            **design, "checkpoint_epoch": epoch, "rollout_seed": rollout_seed,
            "max_rollout_steps": HORIZON_STEPS, "horizon_epochs": 1,
            "coupled_control": False,
        }
        config_digest = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()
        cp_manifest = {
            "schema": "structural_value_pilot_v1",
            "config": config, "config_hash": config_digest,
            "run_id": run_id, "checkpoint_id": cp_id,
            "split": split, "training_population": n_train,
            "rollout_optimizer_steps": HORIZON_STEPS,
            "rollout_endpoints": list(HORIZONS),
            "materialized_batch_sha256": _digest(batches),
            "actions": [a.payload for a in actions],
            "paired_observations": len(observations), "paired_labels": len(labels),
            "test_examples_used_for_selection": 0,
            "interpretation": "G5A same-checkpoint exploratory predictor ablation",
        }
        (cp_dir / "manifest.json").write_text(
            json.dumps(cp_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        audited = audit_pilot(cp_dir)
        (cp_dir / "audit.json").write_text(
            json.dumps(audited, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if (not audited["passed"] or audited["actions"] != 4
                or audited["paired_horizons"] != list(HORIZONS)
                or audited["observations"] != 12):
            raise AssertionError("G5A checkpoint audit failed")
        checkpoints.append({
            "epoch": epoch, "checkpoint_id": cp_id, "pilot_subdir": cp_dir.name,
            "decision_state": state, "proposal_features": features,
            "feature_compute_seconds": feature_seconds,
            "preaction_record": str(cp_dir.name + "/preaction.json"),
            "audited": True,
        })
    report = {
        "schema": "awn_g5a_same_checkpoint_cohort_v1",
        "design": design, "design_hash": design_hash,
        "run_id": run_id, "git_sha": git_sha,
        "dataset": dataset, "seed": seed, "split": split,
        "feature_names": list(FEATURE_NAMES),
        "checkpoints": checkpoints, "baseline_training_history": history,
        "final_horizon_rollouts": len(checkpoints) * 4,
        "all_horizon_observations": len(checkpoints) * 4 * len(HORIZONS),
        "test_examples_used": 0, "passed": len(checkpoints) == 4,
    }
    (output / "g5a_cohort_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"dataset": dataset, "seed": seed, "preaction_checkpoints": len(checkpoints),
                      "final_horizon_rollouts": report["final_horizon_rollouts"],
                      "test_examples_used": 0}, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--seed", choices=SEEDS, type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    run_study(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
