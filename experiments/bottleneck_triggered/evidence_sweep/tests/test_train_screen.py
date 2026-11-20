"""Regression tests for safe AWN calibration and stable neuron identity."""
import types

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.bottleneck_triggered.evidence_sweep.identity import (
    NeuronIdentityTracker, track_awn_width,
)
from experiments.bottleneck_triggered.evidence_sweep.train_screen import (
    safe_loaders, train_epoch, check_validation,
)


def _loaders(test_alias=True):
    base = TensorDataset(torch.arange(100).float().reshape(50, 2),
                         torch.arange(50) % 2)
    train = DataLoader(Subset(base, list(range(30))), batch_size=5, shuffle=True)
    test = DataLoader(Subset(base, list(range(30, 50))), batch_size=5)
    valid = test if test_alias else DataLoader(
        Subset(base, [30, 31, 32]), batch_size=5)
    return train, valid, test


def test_alias_to_test_becomes_train_derived_validation():
    original, validation, test = _loaders()
    train, calibration, val, meta = safe_loaders(
        original, validation, test, seed=42,
        calibration_fraction=0.2, validation_fraction=0.2)
    assert meta["provenance"] == "train_derived_validation"
    assert meta["test_examples_used"] == 0
    assert len(train.dataset) + len(calibration.dataset) + len(val.dataset) == 30
    assert all(len(loader.dataset) > 0 for loader in (train, calibration, val))


def test_train_test_overlap_is_fatal():
    original, valid, _ = _loaders()
    test = DataLoader(Subset(original.dataset.dataset, [7, 8, 9]), batch_size=3)
    with pytest.raises(ValueError, match="train and test"):
        safe_loaders(original, valid, test, seed=1)


def test_true_provided_validation_is_used():
    original, _, _ = _loaders()
    base = original.dataset.dataset
    valid = DataLoader(Subset(base, list(range(30, 35))), batch_size=4)
    test = DataLoader(Subset(base, list(range(35, 50))), batch_size=4)
    train, calibration, val, meta = safe_loaders(original, valid, test, seed=2)
    assert meta["provenance"] == "provided_independent_validation"
    assert len(val.dataset) == 5
    assert len(train.dataset) + len(calibration.dataset) == 30


def test_split_is_reproducible_for_same_seed():
    original, valid, test = _loaders()
    one = safe_loaders(original, valid, test, seed=17)
    two = safe_loaders(original, valid, test, seed=17)
    for idx in (0, 1, 2):
        assert list(one[idx].dataset.indices) == list(two[idx].dataset.indices)


def test_pruned_identity_is_not_reassigned_to_new_neuron():
    tracker = NeuronIdentityTracker("demo", 4)
    initial = tracker.snapshot()
    tracker.resize(2)
    tracker.resize(4)
    final = tracker.snapshot()
    assert final[:2] == initial[:2]
    assert not set(initial[2:]).intersection(final[2:])
    assert len(tracker.events) == 2


class _Dyn:
    num_layers = 1

    def __init__(self):
        self.first = nn.Linear(2, 4)

    def get_layer(self, layer_id):
        assert layer_id == 0
        return self.first

    def change_shape(self, layer_id, num_neurons, change_output):
        assert layer_id == 0 and change_output
        self.first = nn.Linear(2, num_neurons)
        return 1


def test_interception_restores_original_method():
    dyn = _Dyn()
    model = types.SimpleNamespace(dyn_model=dyn)
    with track_awn_width(model, "unit") as tracker:
        old = tracker.snapshot()
        dyn.change_shape(layer_id=0, num_neurons=2, change_output=True)
        dyn.change_shape(layer_id=0, num_neurons=4, change_output=True)
        assert tracker.snapshot()[:2] == old[:2]
        assert tracker.snapshot()[2:] != old[2:]
    assert "change_shape" not in vars(dyn)


class _ClassificationMetric:
    def get_predictions_and_targets(self, y, *outputs):
        return outputs[0], y.reshape(-1).long()

    def compute_metric(self, targets, predictions):
        return nn.functional.cross_entropy(predictions, targets)


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def forward(self, x, skip_update=False):
        return (self.fc(x),)


def test_training_and_validation_pipeline_runs():
    torch.manual_seed(2)
    model = _TinyClassifier()
    loader = DataLoader(TensorDataset(torch.randn(22, 2),
                                      torch.randint(0, 2, (22,))), batch_size=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train = train_epoch(model, loader, optimizer, _ClassificationMetric(),
                        torch.device("cpu"))
    assert train["training_steps"] == 5
    val = check_validation(model, loader, torch.device("cpu"))
    assert val["validation_examples"] == 22
    assert 0 <= val["validation_accuracy"] <= 1
