"""Toy-model tests for full-state replay, matched endpoints and labels.

These tests do not substitute for a native AWN G0 run.
"""
from contextlib import contextmanager
from dataclasses import replace
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.structural_value.actions.action_spec import (
    ActionKind, ActionMeasurement, ActionSpec)
from experiments.structural_value.state.snapshot import TrainingSnapshot
from experiments.structural_value.counterfactual.fork_runner import (
    RolloutObservation, paired_rollouts)
from experiments.structural_value.counterfactual.label_builder import build_labels


class ToyAdapter:
    def __init__(self, runtime):
        self.model = runtime.model
        self.optimizer = runtime.optimizer
        self.device = torch.device("cpu")
        self.x = torch.tensor([[1.0, 0.3], [-0.2, 0.6], [0.4, -0.7]])
        self.y = torch.tensor([0, 1, 0])

    @contextmanager
    def scope(self, gate_mode):
        assert gate_mode in ("positional", "neuron_coupled")
        yield self

    def apply(self, action):
        if action.kind is ActionKind.SKIP:
            return ActionMeasurement(True)
        with torch.no_grad():
            self.model[-1].bias.add_(torch.tensor([0.02, -0.02]))
        return ActionMeasurement(False, evidence_seconds=0.001,
                                 intervention_seconds=0.002)

    def train_step(self, batch):
        x, y = batch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(self.model(x), y).backward()
        self.optimizer.step()

    def validation_risk(self):
        prev = self.model.training
        self.model.eval()
        with torch.no_grad():
            risk = float(F.cross_entropy(self.model(self.x), self.y))
        self.model.train(prev)
        return risk

    def widths(self):
        return (4,)


def fixture_snapshot():
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = opt
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    opt.step()  # Populate Adam exp_avg/exp_avg_sq.
    return TrainingSnapshot.capture(model, opt), (x, y)


def test_snapshot_restores_parameters_optimizer_links_and_global_rng():
    snapshot, batch = fixture_snapshot()
    state_a = snapshot.fork()
    state_b = snapshot.fork()
    assert state_a.model is not state_b.model
    assert state_a.optimizer is not state_b.optimizer
    assert state_a.model.torch_optimizer is state_a.optimizer
    assert set(state_a.model.parameters()) == {
        p for g in state_a.optimizer.param_groups for p in g["params"]}
    for p_a, p_b in zip(state_a.model.parameters(), state_b.model.parameters()):
        assert torch.equal(p_a, p_b)
        assert "exp_avg" in state_a.optimizer.state[p_a]
        assert torch.equal(state_a.optimizer.state[p_a]["exp_avg"],
                           state_b.optimizer.state[p_b]["exp_avg"])
    snapshot.restore_rng()
    v1 = (random.random(), np.random.rand(), torch.rand(1).item())
    snapshot.restore_rng()
    v2 = (random.random(), np.random.rand(), torch.rand(1).item())
    assert v1 == v2


def test_loader_rng_capture_and_restore():
    gen = torch.Generator().manual_seed(90)
    ds = TensorDataset(torch.arange(20))
    loader = DataLoader(ds, batch_size=4, shuffle=True, generator=gen)
    snap, _ = fixture_snapshot()
    captured = TrainingSnapshot.capture(snap.state.model, snap.state.optimizer,
                                        loader=loader)
    target = gen.get_state().clone()
    next(iter(loader))
    assert not torch.equal(gen.get_state(), target)
    captured.restore_rng(loader)
    assert torch.equal(gen.get_state(), target)


def test_skip_skip_is_deterministic_at_each_intermediate_horizon():
    snap, (x, y) = fixture_snapshot()
    batch_plan = ((x[:4], y[:4]), (x[4:], y[4:]), (x[:4], y[:4]))
    actions = [ActionSpec(ActionKind.SKIP)]
    args = dict(run_id="r1", checkpoint_id="r1:step10",
                rollout_seed=50, horizons=[1, 2, 3])
    a = paired_rollouts(snap, ToyAdapter, batch_plan, actions, **args)
    b = paired_rollouts(snap, ToyAdapter, batch_plan, actions, **args)
    assert len(a) == len(b) == 3
    for left, right in zip(a, b):
        assert left.horizon_steps == right.horizon_steps
        assert left.validation_risk == right.validation_risk
        assert left.cost["optimizer_steps"] == left.horizon_steps
        assert left.cost["examples"] == right.cost["examples"]
    assert a[0].validation_risk != a[-1].validation_risk


def test_labels_sign_cost_normalization_and_skip_zero():
    skip = ActionSpec(ActionKind.SKIP)
    action = ActionSpec(ActionKind.WEIGHT_POS)
    base = RolloutObservation(
        "r", "cp", skip, 100, 5, 0.6, 0.7, 0.7,
        {"optimizer_steps": 5, "examples": 50, "online_seconds": 2.0},
        (8,), True)
    treatment = replace(
        base, action=action, validation_risk=0.4,
        cost={"optimizer_steps": 5, "examples": 50, "online_seconds": 2.3},
        no_op=False)
    labels = build_labels([treatment, base],
                          cost_weight=0.5, reference_seconds=1.0)
    a = next(x for x in labels if x.action_id == action.action_id)
    s = next(x for x in labels if x.action_id == skip.action_id)
    assert a.delta_risk == pytest.approx(0.2)
    assert a.delta_online_seconds == pytest.approx(0.3)
    assert a.net_value == pytest.approx(0.05)
    assert (s.delta_risk, s.delta_online_seconds, s.net_value) == (0, 0, 0)
    with pytest.raises(ValueError, match="Unmatched"):
        build_labels([base, replace(treatment,
                         cost={"optimizer_steps": 4, "examples": 50, "online_seconds": 2.3})],
                     cost_weight=0, reference_seconds=1.0)


def test_action_noop_payload_and_invalid_early_labels():
    with pytest.raises(ValueError):
        ActionSpec(ActionKind.RANDOM_POS)
    with pytest.raises(ValueError):
        ActionSpec(ActionKind.SKIP, action_seed=17)
    with pytest.raises(ValueError):
        ActionSpec(ActionKind.WEIGHT_POS, target_widths=(16,))
    with pytest.raises(ValueError):
        ActionSpec(ActionKind.RESIZE)
    assert ActionSpec(ActionKind.GRADIENT_POS).action_id != ActionSpec(ActionKind.WEIGHT_POS).action_id
    snap, batch = fixture_snapshot()
    with pytest.raises(ValueError, match="skip"):
        paired_rollouts(snap, ToyAdapter, (batch,), [ActionSpec(ActionKind.WEIGHT_POS)],
                        [1], run_id="r", checkpoint_id="cp", rollout_seed=0)
    with pytest.raises(ValueError, match="Horizons"):
        paired_rollouts(snap, ToyAdapter, (batch,), [ActionSpec(ActionKind.SKIP)],
                        [2], run_id="r", checkpoint_id="cp", rollout_seed=0)


def test_pairing_rejects_missing_baseline_and_duplicate_observations():
    action = ActionSpec(ActionKind.GRADIENT_POS)
    row = RolloutObservation(
        "r", "cp", action, 1, 1, 0.5, 0.5, 0.5,
        {"optimizer_steps": 1, "examples": 10, "online_seconds": 1.0},
        (4,), False)
    with pytest.raises(ValueError, match="Exactly one"):
        build_labels([row], cost_weight=0, reference_seconds=1)
    skip = replace(row, action=ActionSpec(ActionKind.SKIP), no_op=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build_labels([skip, row, row], cost_weight=0, reference_seconds=1)
