"""Exercise complete controlled runner with a minimal AWN-compatible mock.

This does not claim a real-AWN integration test; the local agent must run
real data and original AWN before interpreting pilot outcomes.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.bottleneck_triggered.evidence_sweep import controlled_bt as bt


class Exponential:
    pass


class Dynamic(nn.Module):
    num_layers = 1
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 5), nn.Linear(5, 2)])
    def get_layer(self, i):
        return self.layers[i]
    def change_shape(self, layer_id, num_neurons, change_output):
        return 0
    def forward(self, x, qW_probs, return_activations=False):
        h = F.leaky_relu(self.layers[0](x))
        h = F.leaky_relu(h * qW_probs[0])
        return self.layers[1](h), h, None


class MockAWN(nn.Module):
    def __init__(self, dim_input_features, dim_target, config):
        super().__init__()
        self.dyn_model = Dynamic()
        self.register_buffer("gates", torch.softmax(-torch.arange(5) * 0.2, dim=0))
        self.variational_Ws = [SimpleNamespace(
            discretized_distribution=SimpleNamespace(base_distribution=Exponential()))]
    def to(self, device):
        super().to(device)
        # The real AWN.to returns None.
    def forward(self, x, skip_update=False):
        return self.dyn_model(x, [self.gates])


def _train(model, loader, optimizer, elbo_fn, device):
    model.train()
    steps, n, total = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x)[0], y)
        loss.backward()
        optimizer.step()
        steps += 1
        n += len(y)
        total += float(loss.detach()) * len(y)
    return {"mean_train_elbo_loss": total / n,
            "training_steps": steps, "training_examples": n}


def test_mock_full_training_pilot_coupled_and_unchanged_baseline(monkeypatch, tmp_path):
    torch.manual_seed(123)
    x = torch.randn(32, 2)
    y = (x[:, 0] + x[:, 1] > 0).long()
    train = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=8)
    cal = DataLoader(TensorDataset(x[16:20], y[16:20]), batch_size=4)
    validation = DataLoader(TensorDataset(x[20:26], y[20:26]), batch_size=6)
    test = DataLoader(TensorDataset(x[26:], y[26:]), batch_size=6)
    fake_common = ModuleType("experiments.bottleneck_triggered.common_utils")
    fake_common.get_awn_config = lambda n_obs, enable_learned_ordering=False: {}
    fake_common.load_dataset_splits = lambda dataset: (train, validation, test, len(train.dataset))
    fake_model = ModuleType("model")
    fake_model.AWN = MockAWN
    fake_metric = ModuleType("metric")
    fake_metric.ELBO_Classification = lambda n_obs: object()
    monkeypatch.setitem(sys.modules, fake_common.__name__, fake_common)
    monkeypatch.setitem(sys.modules, "model", fake_model)
    monkeypatch.setitem(sys.modules, "metric", fake_metric)
    monkeypatch.setattr(bt, "safe_loaders",
                        lambda a, b, c, seed: (
                            train, cal, validation,
                            {"test_examples_used": 0, "train_examples": len(train.dataset)}))
    monkeypatch.setattr(bt, "train_epoch", _train)
    for method, mode in [("identity", "positional"), ("GI", "neuron_coupled"),
                         ("W", "positional")]:
        out = tmp_path / method
        rec = bt.run_experiment(
            "DoubleMoon", 42, method, mode, "fixed", 2, out,
            fixed_epochs=(1,), max_interventions=1,
            max_calibration_batches=1, save_model=True)
        assert rec["optimizer_steps"] == 4
        assert rec["test_examples_used_for_selection"] == 0
        assert (out / "run_manifest.json").is_file()
        assert (out / "final_checkpoint.pt").is_file()
        assert rec["final_width"] == 5
        if method == "identity":
            assert len(rec["interventions"]) == 0
        else:
            assert len(rec["interventions"]) == 1
            if mode == "neuron_coupled":
                assert rec["interventions"][0]["full_width_output_max_diff"] < 1e-4
