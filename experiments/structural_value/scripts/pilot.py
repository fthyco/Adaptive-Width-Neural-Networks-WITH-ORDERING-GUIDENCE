"""Train AWN to a checkpoint, then fork skip/random/W/G at matched steps.

Run from the repository root, in the official AWN environment:
python -m experiments.structural_value.scripts.pilot --dataset Spiral --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..actions.action_spec import ActionKind, ActionSpec
from ..adapters.awn_adapter import AWNAdapter
from ..counterfactual.fork_runner import paired_rollouts
from ..counterfactual.label_builder import build_labels
from ..state.snapshot import TrainingSnapshot


def future_batches(loader: DataLoader, *, seed: int,
                   epochs: int, max_steps: int | None = None
                   ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Freeze the same training-only minibatches for every treatment fork."""
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    result = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed * 1009 + epoch)
        stream = DataLoader(loader.dataset, batch_size=loader.batch_size,
                            shuffle=True, num_workers=0, generator=generator)
        for x, y in stream:
            result.append((x.detach().clone(), y.detach().clone()))
            if max_steps is not None and len(result) >= max_steps:
                return tuple(result)
    return tuple(result)


def _git_sha() -> str:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if len(sha) != 40:
        raise RuntimeError("Unverifiable code commit")
    return sha


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                         encoding="utf-8")
    temporary.replace(path)


def run(*, dataset: str, seed: int, checkpoint_epoch: int, horizon_epochs: int,
        rollout_seed: int, output: Path, device: str, cost_weight: float,
        reference_seconds: float, coupled_control: bool,
        max_rollout_steps: int | None = None) -> dict:
    if dataset not in ("Spiral", "SpiralHard", "DoubleMoon"):
        raise ValueError("Dataset must be from the reference AWN benchmark")
    if checkpoint_epoch < 0 or horizon_epochs < 1:
        raise ValueError("Checkpoint epoch and horizon must be nonnegative/positive")
    if max_rollout_steps is not None and max_rollout_steps < 1:
        raise ValueError("max_rollout_steps must be positive")
    if cost_weight < 0 or reference_seconds <= 0:
        raise ValueError("Invalid cost exchange rate")
    if output.exists():
        raise FileExistsError("Refusing to overwrite or mix immutable pilot output")
    from experiments.bottleneck_triggered.common_utils import (
        get_awn_config, load_dataset_splits)
    from experiments.bottleneck_triggered.evidence_sweep.train_screen import (
        safe_loaders, train_epoch)
    from metric import ELBO_Classification
    from model import AWN

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    original_train, original_val, original_test, _ = load_dataset_splits(dataset)
    train, calibration, validation, split_provenance = safe_loaders(
        original_train, original_val, original_test, seed=seed)
    dev = torch.device(device)
    n_train = len(train.dataset)
    model = AWN(2, 2, get_awn_config(n_train, enable_learned_ordering=False))
    model.to(dev)  # Original AWN .to() need not return self.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = optimizer
    criterion = ELBO_Classification(n_train)
    history = []
    for _ in range(checkpoint_epoch):
        history.append(train_epoch(model, train, optimizer, criterion, dev))
    sha = _git_sha()
    config = {"dataset": dataset, "seed": seed, "checkpoint_epoch": checkpoint_epoch,
              "horizon_epochs": horizon_epochs, "rollout_seed": rollout_seed,
              "device": device, "cost_weight": cost_weight,
              "reference_seconds": reference_seconds,
              "coupled_control": coupled_control,
              "max_rollout_steps": max_rollout_steps, "git_sha": sha,
              "learning_rate": 0.01}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    run_id = f"{dataset}:seed{seed}:{config_hash[:16]}:{sha[:12]}"
    checkpoint_id = f"{run_id}:epoch{checkpoint_epoch}:step{sum(i['training_steps'] for i in history)}"
    snapshot = TrainingSnapshot.capture(
        model, optimizer, loader=train, history=history,
        provenance={"development_split": split_provenance,
                    "run_id": run_id, "checkpoint_id": checkpoint_id})
    batches = future_batches(train, seed=rollout_seed, epochs=horizon_epochs,
                             max_steps=max_rollout_steps)
    if not batches:
        raise RuntimeError("No training batches available for pilot")
    endpoints = sorted({1, max(1, math.ceil(len(batches) / 2)), len(batches)})
    batch_digest = hashlib.sha256()
    for x, y in batches:
        batch_digest.update(x.contiguous().numpy().tobytes())
        batch_digest.update(y.contiguous().numpy().tobytes())
    actions = [
        ActionSpec(ActionKind.SKIP),
        ActionSpec(ActionKind.RANDOM_POS, action_seed=rollout_seed + 301),
        ActionSpec(ActionKind.WEIGHT_POS),
        ActionSpec(ActionKind.GRADIENT_POS),
    ]
    if coupled_control:
        actions.append(ActionSpec(ActionKind.WEIGHT_POS, gate_mode="neuron_coupled"))
    def factory(restored):
        return AWNAdapter(restored, calibration=calibration, validation=validation,
                          train_examples=n_train, device=dev, namespace=checkpoint_id,
                          epoch=checkpoint_epoch)

    rows = paired_rollouts(snapshot, factory, batches, actions, endpoints,
                           run_id=run_id, checkpoint_id=checkpoint_id,
                           rollout_seed=rollout_seed)
    labels = build_labels(rows, cost_weight=cost_weight,
                          reference_seconds=reference_seconds)
    output.mkdir(parents=True, exist_ok=False)
    _atomic_jsonl(output / "observations.jsonl", [r.as_dict() for r in rows])
    _atomic_jsonl(output / "labels.jsonl", [l.as_dict() for l in labels])
    manifest = {"schema": "structural_value_pilot_v1",
                "config": config, "config_hash": config_hash,
                "run_id": run_id, "checkpoint_id": checkpoint_id,
                "split": split_provenance, "training_population": n_train,
                "rollout_optimizer_steps": len(batches),
                "rollout_endpoints": endpoints,
                "materialized_batch_sha256": batch_digest.hexdigest(),
                "actions": [a.payload for a in actions],
                "paired_observations": len(rows),
                "paired_labels": len(labels),
                "test_examples_used_for_selection": 0,
                "interpretation": "Engineering pilot only; not a confirmatory result"}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("Spiral", "SpiralHard", "DoubleMoon"),
                        default="Spiral")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-epoch", type=int, default=10)
    parser.add_argument("--horizon-epochs", type=int, default=5)
    parser.add_argument("--max-rollout-steps", type=int,
                        help="Optional bounded engineering rollout in optimizer steps")
    parser.add_argument("--rollout-seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("outputs/structural_value/pilot_01"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cost-weight", type=float, default=0.0)
    parser.add_argument("--reference-seconds", type=float, default=1.0)
    parser.add_argument("--coupled-control", action="store_true")
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
