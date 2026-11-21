"""G5 exploratory feature-only replay of the original G2 training trajectories.

Recreates *baseline* checkpoints on the original 3 seeds and 2 datasets,
extracting W/G/random candidate descriptors before any action.  Reuses the
already-audited immutable G2 labels only in a separate evaluation job.
No interventions, future rollouts, fitting, final-test examples, or label
access are allowed in this script. A numerical mismatch with the original
G2 checkpoint state fails closed rather than silently pairing unlike states.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from ..g5_evidence_features import FEATURE_NAMES, proposal_features
from ..grid_contract import DATASETS, SEEDS, CHECKPOINTS
from ..state.snapshot import TrainingSnapshot
from .pilot import _git_sha


def replay(*, dataset: str, seed: int, audit_file: Path, output: Path) -> dict:
    if dataset not in DATASETS or seed not in SEEDS:
        raise ValueError("G5 historical replay must use the exact G2 population")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Feature records are immutable")
    original = json.loads(Path(audit_file).read_text(encoding="utf-8"))
    if (original.get("schema") != "awn_structural_value_grid96_per_run_audit_v1"
            or original["dataset"] != dataset or original["seed"] != seed
            or original["git_sha"] != "ffb347214a86e4fbf92912ff2bf4dac54039edd5"
            or original["checkpoint_epochs"] != list(CHECKPOINTS)
            or original["test_examples_used"] != 0
            or not original["passed"]):
        raise AssertionError("Unapproved original G2 checkpoint reference")

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
    if split != original["split"] or split["test_examples_used"] != 0:
        raise AssertionError("Original G2 train/calibration/validation split changed")
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
    history: list[dict] = []
    records = []
    steps = 0
    for epoch in range(1, CHECKPOINTS[-1] + 1):
        record = train_epoch(model, train, optimizer, criterion, device)
        record.update(check_validation(model, validation, device))
        record["epoch"] = epoch
        record["physical_width"] = int(model.dyn_model.get_layer(0).weight.shape[0])
        record["learned_width"] = int(model.variational_Ws[0].compute_truncation_number())
        history.append(record)
        steps += record["training_steps"]
        if epoch not in CHECKPOINTS:
            continue
        source = original["checkpoints"][len(records)]
        previous = history[max(0, epoch - 6)]
        state = {
            "epoch": epoch,
            "physical_width": record["physical_width"],
            "learned_width": record["learned_width"],
            "width_rate": float(model.variational_Ws[0]
                                .discretized_distribution.base_distribution.rate
                                .detach().cpu().item()),
            "train_elbo_loss": record["mean_train_elbo_loss"],
            "validation_ce": record["validation_ce"],
            "validation_accuracy": record["validation_accuracy"],
            "validation_ce_change_last_five_epochs": (
                previous["validation_ce"] - record["validation_ce"]),
            "training_steps_so_far": steps,
        }
        historical_state = source["decision_state"]
        if set(historical_state) != set(state) or source["epoch"] != epoch:
            raise AssertionError("Historical state contract changed")
        for k, value in state.items():
            target = historical_state[k]
            if k in ("epoch", "physical_width", "learned_width", "training_steps_so_far"):
                if value != target:
                    raise AssertionError(f"G2 replay differs in {k} at {epoch}")
            elif not np.isclose(value, target, rtol=1e-5, atol=1e-6):
                raise AssertionError(f"G2 replay differs in {k} at {epoch}: {value} / {target}")

        snapshot = TrainingSnapshot.capture(model, optimizer, loader=train)
        try:
            shadow = snapshot.fork(loader=train).model
            w_scores = evidence_scores(shadow, calibration, "W", device)
            g_scores = evidence_scores(shadow, calibration, "G", device)
            feature_map = proposal_features(
                w_scores=w_scores, g_scores=g_scores,
                learned_width=state["learned_width"],
                random_action_seed=(100000 + seed * 13 + epoch * 997 + 301),
            )
        finally:
            snapshot.restore_rng(loader=train)

        records.append({
            "dataset": dataset, "seed": seed, "epoch": epoch,
            "checkpoint_id": source["checkpoint_id"],
            "reference_decision_state": state,
            "proposal_features": feature_map,
            "no_intervention_applied": True,
        })
    if len(records) != len(CHECKPOINTS):
        raise AssertionError("Feature replay incomplete")
    result = {
        "schema": "awn_g5_predecision_evidence_replay_v1",
        "source_g2_commit": original["git_sha"],
        "source_g2_run": 44848340275,
        "feature_extraction_commit": _git_sha(),
        "dataset": dataset, "seed": seed,
        "feature_names": list(FEATURE_NAMES),
        "checkpoints": records,
        "test_examples_used": 0,
        "future_training_steps_executed": 0,
        "interventions_applied": 0,
        "feature_source": "reference_native_awn_before_action_W_G_and_seeded_random",
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "features.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"dataset": dataset, "seed": seed, "preaction_checkpoints": len(records),
                      "test_examples_used": 0}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    replay(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
