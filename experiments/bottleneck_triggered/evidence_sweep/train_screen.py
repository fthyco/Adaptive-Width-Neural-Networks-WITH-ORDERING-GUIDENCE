"""Train baseline AWN with leakage-free calibration and screening validation.

Exports frozen-checkpoint evidence for Stage 1 (255) or Stage 2 (741), or both
as separately stored reports. Does not train each permutation candidate.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .export_awn import export_checkpoint
from .identity import track_awn_width
from .sweep import run_phase

DEFAULT_EPOCHS = {"DoubleMoon": 30, "Spiral": 50, "SpiralHard": 50}


def _root_indices(dataset) -> tuple[object, set[int]]:
    indices = np.arange(len(dataset), dtype=np.int64)
    current = dataset
    while isinstance(current, Subset):
        indices = np.asarray(current.indices, dtype=np.int64)[indices]
        current = current.dataset
    return current, set(map(int, indices))


def _overlap(left, right) -> bool:
    a, ai = _root_indices(left)
    b, bi = _root_indices(right)
    return a is b and bool(ai & bi)


def safe_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    *,
    seed: int,
    calibration_fraction: float = 0.10,
    validation_fraction: float = 0.15,
    validation_limit: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build strictly disjoint train/calibration/validation subsets.

    Original AWN's BT common_utils substitutes test_set for val_set when the
    provided fold has no validation. That fallback is prohibited for screening.
    """
    if (not 0 < calibration_fraction < 0.5
        or not 0 < validation_fraction < 0.5
        or validation_limit < 1):
        raise ValueError("Invalid split fraction or validation limit")
    original = train_loader.dataset
    if _overlap(original, test_loader.dataset):
        raise ValueError("Original train and test splits overlap")
    val_is_test = (val_loader.dataset is test_loader.dataset
                   or _overlap(val_loader.dataset, test_loader.dataset))
    if not val_is_test and _overlap(original, val_loader.dataset):
        raise ValueError("Original train/validation splits overlap")
    size = len(original)
    if size < 20:
        raise ValueError("At least 20 original training examples are required")
    indices = np.random.default_rng(seed).permutation(size)
    n_cal = max(1, int(round(size * calibration_fraction)))
    if val_is_test:
        n_val = max(1, int(round(size * validation_fraction)))
        validation = Subset(original, indices[:n_val].tolist())
        calibration = Subset(original, indices[n_val:n_val+n_cal].tolist())
        train = Subset(original, indices[n_val+n_cal:].tolist())
        provenance = "train_derived_validation"
    else:
        calibration = Subset(original, indices[:n_cal].tolist())
        train = Subset(original, indices[n_cal:].tolist())
        validation = val_loader.dataset
        provenance = "provided_independent_validation"
    if min(len(train), len(calibration), len(validation)) < 1:
        raise ValueError("Empty training, calibration or validation split")
    pairs = ((train, calibration), (train, validation), (train, test_loader.dataset),
             (calibration, validation), (calibration, test_loader.dataset),
             (validation, test_loader.dataset))
    if any(_overlap(a, b) for a, b in pairs):
        raise ValueError("Split overlap would leak labels between stages")
    if len(validation) > validation_limit:
        rng = np.random.default_rng(seed + 3047)
        selected = sorted(rng.choice(len(validation), validation_limit,
                                     replace=False).tolist())
        validation = Subset(validation, selected)
    batch = train_loader.batch_size or 128
    train_out = DataLoader(train, batch_size=batch, shuffle=True, num_workers=0,
                           generator=torch.Generator().manual_seed(seed + 11))
    calibration_out = DataLoader(calibration, batch_size=batch, shuffle=False,
                                 num_workers=0)
    validation_out = DataLoader(validation, batch_size=min(batch, 256),
                                shuffle=False, num_workers=0)
    return train_out, calibration_out, validation_out, {
        "provenance": provenance, "train_examples": len(train),
        "calibration_examples": len(calibration),
        "screening_validation_examples": len(validation),
        "test_examples_used": 0, "split_seed": seed,
    }


def train_epoch(model, loader: DataLoader, optimizer, elbo_fn, device: torch.device) -> dict:
    """Original AWN ELBO training step, without evidence-driven intervention."""
    model.train()
    total_loss, seen, steps = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        predictions, targets = elbo_fn.get_predictions_and_targets(y, *outputs)
        loss = elbo_fn.compute_metric(targets, predictions)
        if not bool(torch.isfinite(loss).all()):
            raise ValueError("Nonfinite AWN ELBO loss")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(x)
        seen += len(x)
        steps += 1
    return {"mean_train_elbo_loss": total_loss / max(1, seen),
            "training_steps": steps, "training_examples": seen}


def check_validation(model, loader: DataLoader, device: torch.device) -> dict:
    """Held-out task CE and accuracy, not ELBO; no resizing during evaluation."""
    correct, seen, total_ce = 0, 0, 0.0
    original_mode = model.training
    try:
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device).reshape(-1).long()
                logits = model(x, skip_update=True)[0]
                total_ce += float(F.cross_entropy(logits, y, reduction="sum"))
                correct += int((logits.argmax(dim=-1) == y).sum())
                seen += len(y)
    finally:
        model.train(original_mode)
    return {"validation_ce": total_ce / max(1, seen),
            "validation_accuracy": correct / max(1, seen),
            "validation_examples": seen}


def run_training(
    dataset: str, seed: int, *, epochs: int, checkpoint_every: int,
    phase: str, output: Path, device: str = "cpu", learning_rate: float = 0.01,
    validation_limit: int = 256, save_model: bool = False,
    max_calibration_batches: int | None = None,
) -> dict:
    """Train one AWN baseline and screen the declared phase at checkpoints."""
    if dataset not in DEFAULT_EPOCHS or epochs < 1 or checkpoint_every < 1:
        raise ValueError("Invalid dataset, epoch count or checkpoint cadence")
    if phase not in ("1", "2", "both"):
        raise ValueError("phase must be 1, 2, or both")
    if max_calibration_batches is not None and max_calibration_batches < 1:
        raise ValueError("Invalid calibration batch limit")

    from experiments.bottleneck_triggered.common_utils import (
        get_awn_config, load_dataset_splits,
    )
    from metric import ELBO_Classification
    from model import AWN

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    raw_train, raw_val, raw_test, _ = load_dataset_splits(dataset)
    train, calibration, validation, split_audit = safe_loaders(
        raw_train, raw_val, raw_test, seed=seed,
        validation_limit=validation_limit,
    )
    if max_calibration_batches is not None:
        from itertools import islice
        calibration = list(islice(calibration, max_calibration_batches))
        if not calibration:
            raise ValueError("No examples in calibration window")
    cfg = get_awn_config(len(train.dataset), enable_learned_ordering=False)
    model = AWN(dim_input_features=2, dim_target=2, config=cfg).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.torch_optimizer = optimizer
    elbo_fn = ELBO_Classification(len(train.dataset))
    output.mkdir(parents=True, exist_ok=True)
    records = []
    history: list[tuple[tuple[str, ...], Path]] = []
    namespace = f"{dataset}:seed{seed}"
    start = time.perf_counter()
    with track_awn_width(model, namespace) as tracker:
        for epoch in range(1, epochs + 1):
            record = train_epoch(model, train, optimizer, elbo_fn, dev)
            record.update(check_validation(model, validation, dev))
            physical_width = int(model.dyn_model.get_layer(0).weight.shape[0])
            ids = tracker.snapshot()
            if physical_width != len(ids):
                raise RuntimeError("Stable ID and physical width mismatch")
            record.update({"epoch": epoch, "physical_width": physical_width})
            if epoch % checkpoint_every == 0 or epoch == epochs:
                stem = f"epoch{epoch:04d}"
                evidence_path = output / f"{stem}_evidence.npz"
                validation_path = output / f"{stem}_validation.npz"
                prior = [path for prior_ids, path in history[-2:]
                         if prior_ids == ids]
                if len(prior) != 2:
                    prior = []
                started = time.perf_counter()
                export_checkpoint(
                    model, calibration, validation, ids,
                    evidence_path, validation_path,
                    checkpoint_id=f"{namespace}:{stem}",
                    prior_evidence_paths=prior,
                )
                record["evidence_seconds"] = time.perf_counter() - started
                record["feature_path"] = str(evidence_path)
                if save_model:
                    checkpoint_path = output / f"{stem}_model.pt"
                    torch.save({
                        "epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "stable_ids": ids, "dataset": dataset, "seed": seed,
                        "config": cfg, "train_split": split_audit,
                    }, checkpoint_path)
                    record["model_checkpoint"] = str(checkpoint_path)
                for stage in ((1, 2) if phase == "both" else (int(phase),)):
                    result_path = output / f"{stem}_phase{stage}.json"
                    started = time.perf_counter()
                    result = run_phase(stage, evidence_path, result_path,
                                       validation_path)
                    record[f"phase{stage}"] = {
                        "report": str(result_path),
                        "planned": result["planned"],
                        "ranked": result["ranked"],
                        "skipped": result["skipped"],
                        "seconds": time.perf_counter() - started,
                    }
                history.append((ids, evidence_path))
                history = history[-2:]
            records.append(record)
        events = tracker.events
        active_ids = tracker.snapshot()
    manifest = {
        "schema_version": "awn_two_phase_real_checkpoint_v1",
        "dataset": dataset, "seed": seed, "epochs": epochs,
        "checkpoint_every": checkpoint_every, "fusion_phase": phase,
        "device": str(dev), "learning_rate": learning_rate,
        "data_contract": split_audit,
        "elapsed_seconds": time.perf_counter() - start,
        "measurement_semantics": "observational AWN checkpoint ranking; no BT intervention",
        "active_stable_ids": active_ids, "identity_events": events,
        "epoch_reports": records,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_EPOCHS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--phase", choices=("1", "2", "both"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--validation-limit", type=int, default=256)
    parser.add_argument("--max-calibration-batches", type=int)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args(argv)
    result = run_training(
        args.dataset, args.seed, epochs=args.epochs or DEFAULT_EPOCHS[args.dataset],
        checkpoint_every=args.checkpoint_every, phase=args.phase,
        output=args.output, device=args.device, learning_rate=args.learning_rate,
        validation_limit=args.validation_limit,
        max_calibration_batches=args.max_calibration_batches,
        save_model=args.save_model,
    )
    print(json.dumps({key: result[key] for key in (
        "dataset", "seed", "epochs", "fusion_phase", "elapsed_seconds",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
