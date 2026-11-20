"""Local CPU tests for identity-coupled live BT physical reordering."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.bottleneck_triggered.evidence_sweep.identity import NeuronIdentityTracker
from experiments.bottleneck_triggered.evidence_sweep.controlled_bt import (
    CoupledIdentity, gate_control, apply_intervention, candidate_order,
    evidence_scores, _finite_scores,
)


class Exponential:
    pass


class ToyDynamic(nn.Module):
    num_layers = 1

    def __init__(self, width=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, width), nn.Linear(width, 2)])
        self.activation_outer = F.leaky_relu

    def get_layer(self, i):
        return self.layers[i]

    def forward(self, x, qW_probs, return_activations=False):
        h = F.relu(self.layers[0](x))
        h = F.leaky_relu(h * qW_probs[0])
        return self.layers[1](h), h, None


class ToyAWN(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.dyn_model = ToyDynamic(width)
        self.register_buffer("gate", torch.softmax(-0.3 * torch.arange(width), 0))
        self.variational_Ws = [SimpleNamespace(
            discretized_distribution=SimpleNamespace(base_distribution=Exponential()))]

    def forward(self, x, skip_update=False):
        return self.dyn_model(x, [self.gate])


def _data(n=12):
    x = torch.randn(n, 3)
    y = torch.randint(0, 2, (n,))
    return x, y


def test_live_coupled_gate_permutation_preserves_output_and_adam_states():
    torch.manual_seed(13)
    model = ToyAWN()
    x, y = _data()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    opt.zero_grad()
    F.cross_entropy(model(x)[0], y).backward()
    opt.step()
    first = model.dyn_model.get_layer(0)
    head = model.dyn_model.get_layer(1)
    incoming_before = first.weight.detach().clone()
    outgoing_before = head.weight.detach().clone()
    state_in_before = opt.state[first.weight]["exp_avg"].clone()
    state_out_before = opt.state[head.weight]["exp_avg_sq"].clone()
    tracker = NeuronIdentityTracker("test", 4)
    state = CoupledIdentity(tracker)
    order = [3, 1, 0, 2]
    try:
        with gate_control(model, state, "neuron_coupled"):
            before = model(x, skip_update=True)[0].detach().clone()
            info = apply_intervention(model, opt, state, order, x,
                                      "neuron_coupled", epoch=4, method="G")
            after = model(x, skip_update=True)[0].detach().clone()
            assert (before - after).abs().max().item() < 1e-5
            assert info["full_width_output_max_diff"] < 1e-5
            assert state.permutation_ranks() == order
            assert tracker.snapshot() == tuple("test:neuron:%06d" % i for i in order)
            assert torch.allclose(first.weight, incoming_before[order])
            assert torch.allclose(head.weight, outgoing_before[:, order])
            assert torch.allclose(opt.state[first.weight]["exp_avg"],
                                  state_in_before[order])
            assert torch.allclose(opt.state[head.weight]["exp_avg_sq"],
                                  state_out_before[:, order])
    finally:
        state.close()


def test_live_positional_gate_permutation_can_change_function():
    torch.manual_seed(18)
    model = ToyAWN()
    x, _ = _data()
    tracker = NeuronIdentityTracker("pos", 4)
    state = CoupledIdentity(tracker)
    try:
        with gate_control(model, state, "positional"):
            result = apply_intervention(model, None, state, [3, 1, 0, 2], x,
                                        "positional", epoch=4, method="W")
            assert result["no_gate_output_max_diff"] < 1e-5
            assert result["full_width_output_max_diff"] > 1e-6
    finally:
        state.close()


def test_identity_tracker_survivors_keep_relative_gate_priority():
    tracker = NeuronIdentityTracker("resize", 5)
    state = CoupledIdentity(tracker)
    try:
        state.permute([4, 1, 3, 0, 2])
        assert state.permutation_ranks() == [4, 1, 3, 0, 2]
        state.resize(3)
        assert state.permutation_ranks() == [2, 0, 1]
        remaining = tracker.snapshot()
        state.resize(5)
        assert tracker.snapshot()[:3] == remaining
        assert len(set(tracker.snapshot())) == 5
        assert state.permutation_ranks() == [2, 0, 1, 3, 4]
    finally:
        state.close()


def test_gate_override_removed_after_scope():
    model = ToyAWN()
    tracker = NeuronIdentityTracker("test", 4)
    state = CoupledIdentity(tracker)
    assert "forward" not in vars(model.dyn_model)
    with gate_control(model, state, "neuron_coupled"):
        assert "forward" in vars(model.dyn_model)
    assert "forward" not in vars(model.dyn_model)
    state.close()


def test_evidence_g_i_and_gi_are_finite_and_deterministic():
    torch.manual_seed(7)
    model = ToyAWN()
    x, y = _data(n=16)
    data = DataLoader(TensorDataset(x, y), batch_size=8)
    g = evidence_scores(model, data, "G", torch.device("cpu"))
    i = evidence_scores(model, data, "I", torch.device("cpu"))
    gi = evidence_scores(model, data, "GI", torch.device("cpu"))
    assert all(v.shape == (4,) and np.isfinite(v).all() for v in (g, i, gi))
    assert np.array_equal(g, evidence_scores(model, data, "G", torch.device("cpu")))
    assert np.array_equal(i, evidence_scores(model, data, "I", torch.device("cpu")))


def test_canonical_tie_break_and_tie_shuffle_preserve_permutation():
    ids = ("id:004", "id:001", "id:003", "id:002")
    scores = np.array([2., 2., 1., 1.])
    expected = [1, 0, 3, 2]
    assert candidate_order(scores, ids, "S", 42) == expected
    shuffle_a = candidate_order(scores, ids, "S_tie_shuffle", 42)
    shuffle_b = candidate_order(scores, ids, "S_tie_shuffle", 42)
    assert shuffle_a == shuffle_b
    assert sorted(shuffle_a) == list(range(4))
    assert set(shuffle_a[:2]) == {0, 1}
    assert set(shuffle_a[2:]) == {2, 3}


def test_no_score_imputation():
    with pytest.raises(ValueError):
        _finite_scores(np.array([1., np.nan]), 2)
    with pytest.raises(ValueError):
        _finite_scores(np.array([1.]), 2)
