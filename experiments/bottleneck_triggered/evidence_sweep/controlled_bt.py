"""Controlled, real-training AWN bottleneck interventions (one-hidden-layer MLP).

This is an experimental extension. It never edits original AWN modules.
Neuron-coupled gates are a counterfactual, not the original AWN variational
model: after width changes the surviving gate priorities are re-ranked to the
current support of the width distribution.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .sweep import fuse
from .identity import track_awn_width
from .train_screen import safe_loaders, train_epoch, check_validation, DEFAULT_EPOCHS

METHODS = ("identity", "random", "W", "G", "J", "I", "GI", "S", "S_tie_shuffle")
GATE_MODES = ("positional", "neuron_coupled")


class CoupledIdentity:
    """Track stable neuron IDs and each identity's relative gate priority.

    When width shrinks, surviving priorities are compacted before gate lookup.
    When width grows, newly created neurons receive later priorities. The
    mapping is changed only AFTER physical permutation of weights/optimizer.
    """
    def __init__(self, tracker):
        self.tracker = tracker
        self.priorities = list(range(len(tracker.ids)))
        self.next_priority = len(self.priorities)
        self._resize = tracker.resize
        tracker.resize = self.resize

    def resize(self, width: int) -> None:
        old = len(self.priorities)
        self._resize(width)
        if width < old:
            self.priorities = self.priorities[:width]
        elif width > old:
            self.priorities.extend(range(self.next_priority, self.next_priority + width - old))
            self.next_priority += width - old
        self.assert_aligned()

    def assert_aligned(self) -> None:
        if (len(self.priorities) != len(self.tracker.ids)
                or len(set(self.priorities)) != len(self.priorities)
                or len(set(self.tracker.ids)) != len(self.tracker.ids)):
            raise RuntimeError("Neuron identity/gate-priority alignment lost")

    def permutation_ranks(self) -> list[int]:
        self.assert_aligned()
        sorted_slots = sorted(range(len(self.priorities)), key=lambda i: self.priorities[i])
        ranks = [0] * len(sorted_slots)
        for rank, slot in enumerate(sorted_slots):
            ranks[slot] = rank
        return ranks

    def permute(self, perm: list[int]) -> None:
        n = len(self.priorities)
        if sorted(perm) != list(range(n)):
            raise ValueError("Not a complete physical neuron permutation")
        self.tracker.ids[:] = [self.tracker.ids[i] for i in perm]
        self.priorities = [self.priorities[i] for i in perm]
        self.assert_aligned()

    def close(self) -> None:
        self.tracker.resize = self._resize


@contextmanager
def gate_control(model, identity: CoupledIdentity, mode: str):
    """Change only the gate vector delivered to DynamicMLP.forward.

    Supports exponential AWN gates; qW_probs remains differentiable in rate.
    In coupled mode the vector is indexed by identity gate priority instead of
    physical slot. No new trainable parameter or KL term is introduced.
    """
    if mode not in GATE_MODES:
        raise ValueError("Unknown gating mode")
    dyn = model.dyn_model
    original = dyn.forward
    had_override = "forward" in vars(dyn)
    if mode == "neuron_coupled":
        dist = model.variational_Ws[0].discretized_distribution.base_distribution
        if dist.__class__.__name__ != "Exponential":
            raise ValueError("Coupled gating is implemented only for exponential AWN")
        def controlled_forward(data, qW_probs, return_activations=False):
            if len(qW_probs) != 1:
                raise ValueError("Only one-hidden-layer AWN is supported")
            g = qW_probs[0]
            ranks = identity.permutation_ranks()
            if g.numel() != len(ranks):
                raise RuntimeError("Gate vector and active neuron identities mismatch")
            idx = torch.tensor(ranks, dtype=torch.long, device=g.device)
            return original(data, [g.index_select(0, idx)],
                            return_activations=return_activations)
        dyn.forward = controlled_forward
    try:
        yield
    finally:
        if mode == "neuron_coupled":
            if had_override:
                dyn.forward = original
            else:
                delattr(dyn, "forward")


def _finite_scores(scores: np.ndarray, width: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (width,) or not np.isfinite(scores).all():
        raise ValueError("Missing/nonfinite neuron evidence or invalid width")
    return scores


def evidence_scores(model, calibration, method: str, device: torch.device) -> np.ndarray:
    """Real AWN-derived evidence, with task-CE attribution and no optimizer step.

    G = mean per-example abs gradient dCE/d(head input).
    J = mean abs(head input * per-example gradient).
    I = exact linear-head single-neuron masking delta CE, vectorized.
    S = GRAPHyco structural perturbation proxy, NOT causal ablation.
    """
    if method not in METHODS or method in ("identity", "random"):
        raise ValueError("Method has no evidence scores")
    dyn = model.dyn_model
    if dyn.num_layers != 1:
        raise ValueError("Only AWN one-hidden-layer DynamicMLP supported")
    first, head = dyn.get_layer(0), dyn.get_layer(1)
    width = first.weight.shape[0]
    if not isinstance(first, torch.nn.Linear) or not isinstance(head, torch.nn.Linear):
        raise ValueError("Expected two linear AWN layers")
    wi = first.weight.detach()
    wo = head.weight.detach()
    if method == "W":
        score = torch.linalg.vector_norm(wi, dim=1) * torch.linalg.vector_norm(wo, dim=0)
        return _finite_scores(score.cpu().numpy(), width)
    if method in ("S", "S_tie_shuffle"):
        from .export_awn import _static_graph_scores
        return _finite_scores(
            _static_graph_scores(wi.cpu().numpy(), wo.cpu().numpy(), 0.05), width
        )

    sums = {k: np.zeros(width, dtype=np.float64) for k in ("G", "J", "I")}
    need_grad = method in ("G", "J", "GI")
    need_masking = method in ("I", "GI")
    n = 0
    captured = {}
    def remember_input(_module, args):
        captured["head_input"] = args[0]
    hook = head.register_forward_pre_hook(remember_input)
    old_mode = model.training
    try:
        model.eval()
        with (torch.enable_grad() if need_grad else torch.no_grad()):
            for x, y in calibration:
                x = x.to(device)
                y = y.to(device).reshape(-1).long()
                b = len(y)
                logits = model(x, skip_update=True)[0]
                h = captured["head_input"]
                if h.shape != (b, width):
                    raise RuntimeError("Unexpected hidden activation dimensions")
                grad = (torch.autograd.grad(F.cross_entropy(logits, y), h)[0] * b
                        if need_grad else None)
                with torch.no_grad():
                    if need_grad:
                        sums["G"] += grad.abs().double().sum(0).cpu().numpy()
                        sums["J"] += (h.detach() * grad).abs().double().sum(0).cpu().numpy()
                    if need_masking:
                        contributions = h.detach().unsqueeze(-1) * wo.T.unsqueeze(0)
                        removed = logits.detach().unsqueeze(1) - contributions
                        ce_removed = F.cross_entropy(
                            removed.permute(0, 2, 1), y[:, None].expand(b, width),
                            reduction="none")
                        ce_original = F.cross_entropy(logits.detach(), y, reduction="none")
                        sums["I"] += (ce_removed - ce_original[:, None]).double().sum(0).cpu().numpy()
                n += b
    finally:
        hook.remove()
        model.train(old_mode)
    if n < 1:
        raise ValueError("Empty calibration set")
    for key in sums:
        sums[key] = _finite_scores(sums[key] / n, width)
    if method == "GI":
        return _finite_scores(fuse(("G", "I"), "mean_percentile", sums), width)
    return sums[method]


def candidate_order(scores: np.ndarray, stable_ids: tuple[str, ...],
                    method: str, seed: int) -> list[int]:
    """Stable-ID tie breaking; deterministic tie-shuffling control for S."""
    n = len(stable_ids)
    score = _finite_scores(scores, n)
    order = sorted(range(n), key=lambda i: (-float(score[i]), stable_ids[i]))
    if method == "S_tie_shuffle":
        rng = np.random.default_rng(seed)
        start = 0
        while start < n:
            end = start + 1
            while end < n and score[order[end]] == score[order[start]]:
                end += 1
            block = order[start:end]
            rng.shuffle(block)
            order[start:end] = block
            start = end
    return order


def apply_intervention(model, optimizer, identity: CoupledIdentity, order: list[int],
                       sample_x, gate_mode: str, epoch: int, method: str,
                       tolerance: float = 1e-4) -> dict:
    """Migrate weights, biases, Adam states and stable IDs together."""
    from experiments.bottleneck_triggered.permutation.permutation_manager import PermutationManager
    n = len(identity.tracker.ids)
    if sorted(order) != list(range(n)):
        raise ValueError("Invalid full permutation")
    before_ids = tuple(identity.tracker.ids)
    before_priorities = list(identity.priorities)
    old_mode = model.training
    model.eval()
    with torch.no_grad():
        before = model(sample_x, skip_update=True)[0].detach().clone()
    # The existing manager accepts scores rather than an explicit permutation.
    # Unique artificial ranks represent exactly the preselected permutation.
    rank_scores = torch.empty(n, dtype=torch.float32, device=sample_x.device)
    rank_scores[torch.as_tensor(order, device=sample_x.device)] = torch.arange(
        n, 0, -1, dtype=torch.float32, device=sample_x.device)
    manager = PermutationManager(tolerance=tolerance)
    record = manager.permute_layer(
        model, layer_id=0, scores=rank_scores, optimizer=optimizer,
        epoch=epoch, method_name=method, val_sample_x=sample_x)
    if list(record.new_permutation) != order:
        raise RuntimeError("Permutation manager applied an unexpected order")
    identity.permute(order)
    with torch.no_grad():
        after = model(sample_x, skip_update=True)[0].detach()
        max_diff = float((after - before).abs().max())
    model.train(old_mode)
    if gate_mode == "neuron_coupled" and max_diff > tolerance:
        raise RuntimeError("Coupled-gate full-width invariance failed: %.8g" % max_diff)
    return {
        "epoch": epoch, "method": method, "gate_mode": gate_mode,
        "old_ids": list(before_ids), "new_ids": list(identity.tracker.ids),
        "old_priorities": before_priorities, "new_priorities": identity.priorities.copy(),
        "permutation": order, "full_width_output_max_diff": max_diff,
        "no_gate_output_max_diff": float(record.max_output_diff),
        "permutation_seconds": record.reordering_duration_sec,
        "optimizer_migration_seconds": record.optimizer_migration_duration_sec,
        "manager_intermediate_awn_diff_is_pre_priority_update": True,
    }


def run_experiment(dataset: str, seed: int, method: str, gate_mode: str,
                   schedule: str, epochs: int, output: Path,
                   fixed_epochs: tuple[int, ...] = (10, 20),
                   max_interventions: int = 2, device: str = "cpu",
                   learning_rate: float = 0.01,
                   max_calibration_batches: int | None = None,
                   save_model: bool = False) -> dict:
    """Train ONE controlled AWN model with predeclared intervention logic.

    All compared methods must use identical data split, seed, epoch budget, LR,
    and schedule. Fixed schedule isolates ordering from adaptive trigger timing.
    Test split is evaluated only once after the full training run.
    """
    if method not in METHODS or gate_mode not in GATE_MODES:
        raise ValueError("Unknown method or gating mode")
    if schedule not in ("fixed", "bt") or epochs < 1 or max_interventions < 0:
        raise ValueError("Invalid schedule or training parameters")
    if any(e < 1 for e in fixed_epochs) or len(set(fixed_epochs)) != len(fixed_epochs):
        raise ValueError("fixed_epochs must contain unique positive integers")
    if max_calibration_batches is not None and max_calibration_batches < 1:
        raise ValueError("max_calibration_batches must be positive")
    from experiments.bottleneck_triggered.common_utils import get_awn_config, load_dataset_splits
    from experiments.bottleneck_triggered.methods.bottleneck_detector import BottleneckDetector
    from metric import ELBO_Classification
    from model import AWN

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    raw_train, raw_val, raw_test, _ = load_dataset_splits(dataset)
    train, calibration, validation, split = safe_loaders(
        raw_train, raw_val, raw_test, seed=seed)
    if max_calibration_batches is not None:
        from itertools import islice
        calibration = list(islice(calibration, max_calibration_batches))
        if not calibration:
            raise ValueError("Calibration window empty")
    cfg = get_awn_config(len(train.dataset), enable_learned_ordering=False)
    model = AWN(dim_input_features=2, dim_target=2, config=cfg)
    model.to(dev)  # AWN.to is not chainable
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.torch_optimizer = optimizer
    elbo_fn = ELBO_Classification(len(train.dataset))
    fixed = set(fixed_epochs)
    detector = BottleneckDetector(
        plateau_window=3, min_relative_improvement=0.005,
        patience=2, min_progress_epochs=6,
        min_interval_between_interventions=5,
        max_interventions=max_interventions)
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    epochs_log, interventions = [], []
    namespace = "%s:seed%d:%s:%s" % (dataset, seed, method, gate_mode)
    with track_awn_width(model, namespace) as tracker:
        identity = CoupledIdentity(tracker)
        try:
            with gate_control(model, identity, gate_mode):
                for epoch in range(1, epochs + 1):
                    train_info = train_epoch(model, train, optimizer, elbo_fn, dev)
                    val_info = check_validation(model, validation, dev)
                    should_trigger, diagnostic = detector.record_and_evaluate(
                        epoch, float(val_info["validation_ce"]))
                    scheduled = ((epoch in fixed) if schedule == "fixed" else should_trigger)
                    if method == "identity":
                        scheduled = False
                    if len(interventions) >= max_interventions:
                        scheduled = False
                    info = {"epoch": epoch, **train_info, **val_info,
                            "width": len(tracker.ids),
                            "gate_priorities": identity.priorities.copy(),
                            "bottleneck_decision": diagnostic.decision}
                    if scheduled:
                        t_evidence = time.perf_counter()
                        if method == "random":
                            order = np.random.default_rng(seed * 10007 + epoch).permutation(
                                len(tracker.ids)).tolist()
                            evidence = None
                        else:
                            evidence = evidence_scores(model, calibration, method, dev)
                            order = candidate_order(evidence, tracker.snapshot(), method,
                                                    seed * 10007 + epoch)
                        evidence_seconds = time.perf_counter() - t_evidence
                        x_verify = next(iter(validation))[0].to(dev)
                        intervention = apply_intervention(
                            model, optimizer, identity, order, x_verify,
                            gate_mode, epoch, method)
                        intervention["evidence_seconds"] = evidence_seconds
                        intervention["score"] = evidence.tolist() if evidence is not None else None
                        interventions.append(intervention)
                        info["intervention"] = len(interventions)
                    epochs_log.append(info)
                # Test evaluation is post-training only; never fed to detector/scorer.
                test_loader = DataLoader(raw_test.dataset, batch_size=256,
                                         shuffle=False, num_workers=0)
                test_result = check_validation(model, test_loader, dev)
                if save_model:
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "neuron_ids": tracker.snapshot(),
                        "gate_priorities": identity.priorities.copy(),
                        "dataset": dataset, "seed": seed,
                        "method": method, "gate_mode": gate_mode,
                        "schedule": schedule, "epochs": epochs,
                    }, output / "final_checkpoint.pt")
        finally:
            identity.close()
    elapsed = time.perf_counter() - started
    try:
        source_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_sha = "unknown"
    manifest = {
        "schema": "controlled_awn_bt_v1",
        "git_sha": source_sha, "save_model": bool(save_model),
        "checkpoint_path": str(output / "final_checkpoint.pt") if save_model else None,
        "active_neuron_ids": tracker.snapshot(),
        "active_gate_priorities": identity.priorities.copy(),
        "max_calibration_batches": max_calibration_batches,
        "dataset": dataset, "seed": seed, "method": method,
        "gate_mode": gate_mode, "schedule": schedule, "epochs": epochs,
        "fixed_epochs": sorted(fixed), "max_interventions": max_interventions,
        "learning_rate": learning_rate, "device": str(dev),
        "split": split, "test_examples_used_for_selection": 0,
        "test_evaluated_after_training": True,
        "experiment_scope": "live_training_intervention_not_frozen_screening",
        "coupled_gates_note": "relative neuron priority over current width support; counterfactual",
        "optimizer_steps": sum(x["training_steps"] for x in epochs_log),
        "training_examples": sum(x["training_examples"] for x in epochs_log),
        "elapsed_seconds": elapsed,
        "epochs_log": epochs_log, "interventions": interventions,
        "final_width": len(tracker.ids),
        "final_test_accuracy": test_result["validation_accuracy"],
        "final_test_ce": test_result["validation_ce"],
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_EPOCHS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--gate-mode", choices=GATE_MODES, required=True)
    parser.add_argument("--schedule", choices=("fixed", "bt"), default="fixed")
    parser.add_argument("--fixed-epochs", default="10,20")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-interventions", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-calibration-batches", type=int)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    fixed = tuple(int(x) for x in args.fixed_epochs.split(",") if x.strip())
    result = run_experiment(
        dataset=args.dataset, seed=args.seed, method=args.method,
        gate_mode=args.gate_mode, schedule=args.schedule,
        epochs=args.epochs or DEFAULT_EPOCHS[args.dataset], output=args.output,
        fixed_epochs=fixed, max_interventions=args.max_interventions,
        device=args.device, learning_rate=args.learning_rate,
        max_calibration_batches=args.max_calibration_batches,
        save_model=args.save_model)
    print(json.dumps({k: result[k] for k in (
        "dataset", "seed", "method", "gate_mode", "schedule",
        "optimizer_steps", "final_width", "final_test_accuracy",
        "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
