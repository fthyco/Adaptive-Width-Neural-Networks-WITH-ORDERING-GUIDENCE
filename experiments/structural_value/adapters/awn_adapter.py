"""AWN one-hidden-layer DynamicMLP adapter, reusing validated BT operations.

The original model, distribution, optimizer and ELBO modules are unchanged.
Requires the normal AWN research environment (including mlwiz/torch-geometric).
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from ..actions.action_spec import ActionKind, ActionMeasurement, ActionSpec
from ..state.snapshot import ForkState


class AWNAdapter:
    def __init__(self, runtime: ForkState, *, calibration, validation,
                 train_examples: int, device: torch.device, namespace: str,
                 epoch: int = 0):
        if not namespace or train_examples < 1:
            raise ValueError("Missing AWN run identity/training population")
        from metric import ELBO_Classification
        self.device = device
        self.model, self.optimizer = runtime.model, runtime.optimizer
        self.model.torch_optimizer = self.optimizer
        self.calibration, self.validation = calibration, validation
        self.criterion = ELBO_Classification(train_examples)
        self.namespace, self.epoch = namespace, epoch
        self.identity = None
        if self.model.dyn_model.num_layers != 1:
            raise ValueError("Pilot requires one-hidden-layer AWN DynamicMLP")

    @contextmanager
    def scope(self, gate_mode: str):
        from experiments.bottleneck_triggered.evidence_sweep.identity import track_awn_width
        from experiments.bottleneck_triggered.evidence_sweep.controlled_bt import (
            CoupledIdentity, gate_control)
        with track_awn_width(self.model, self.namespace) as tracker:
            identity = CoupledIdentity(tracker)
            self.identity = identity
            try:
                with gate_control(self.model, identity, gate_mode):
                    yield self
            finally:
                identity.close()
                self.identity = None

    def validation_risk(self) -> float:
        """Predictive CE on development validation; NOT the AWN training ELBO."""
        previous = self.model.training
        self.model.eval()
        total, count = 0.0, 0
        try:
            with torch.no_grad():
                for x, y in self.validation:
                    x = x.to(self.device)
                    y = y.to(self.device).reshape(-1).long()
                    logits = self.model(x, skip_update=True)[0]
                    total += float(F.cross_entropy(logits, y, reduction="sum"))
                    count += len(x)
        finally:
            self.model.train(previous)
        if count == 0:
            raise ValueError("Development validation set is empty")
        return total / count

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        """One original AWN ELBO update. Native AWN width changes remain enabled."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        if y.ndim == 1:
            y = y.unsqueeze(1)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(x)
        pred, targets = self.criterion.get_predictions_and_targets(y, *outputs)
        loss = self.criterion.compute_metric(targets, pred)
        if not bool(torch.isfinite(loss).all()):
            raise ValueError("Nonfinite AWN ELBO")
        loss.backward()
        self.optimizer.step()

    def widths(self) -> tuple[int, ...]:
        return (int(self.model.variational_Ws[0].compute_truncation_number()),)

    def apply(self, action: ActionSpec) -> ActionMeasurement:
        if action.kind is ActionKind.SKIP:
            return ActionMeasurement(no_op=True)
        from experiments.bottleneck_triggered.evidence_sweep.controlled_bt import (
            apply_intervention, candidate_order, evidence_scores)
        if action.kind in (ActionKind.RESIZE, ActionKind.JOINT):
            raise NotImplementedError("Resize and joint are reserved for G4/G5")
        if self.identity is None:
            raise RuntimeError("Enter AWN gate/identity scope before applying an action")
        ids = self.identity.tracker.snapshot()
        n = len(ids)
        if n < 2:
            return ActionMeasurement(no_op=True)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        if action.kind is ActionKind.RANDOM_POS:
            order = np.random.default_rng(action.action_seed).permutation(n).tolist()
            method = "random"
        else:
            method = "W" if action.kind is ActionKind.WEIGHT_POS else "G"
            scores = evidence_scores(self.model, self.calibration, method, self.device)
            order = candidate_order(scores, ids, method, action.action_seed or 0)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        evidence_seconds = perf_counter() - started
        if order == list(range(n)):
            return ActionMeasurement(True, evidence_seconds, details={"permutation": order})
        sample_x = next(iter(self.validation))[0].to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = perf_counter()
        detail = apply_intervention(
            self.model, self.optimizer, self.identity, order,
            sample_x, action.gate_mode, epoch=self.epoch, method=method)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return ActionMeasurement(
            False, evidence_seconds, perf_counter() - started, details=detail)
