"""End-to-end 255+741 test with measured PyTorch toy activations and gradients.

Static graph signal is deliberately stubbed because this is a test fixture,
not a certification of installed GRAPHyco or actual trained AWN performance.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.bottleneck_triggered.evidence_sweep import export_awn
from experiments.bottleneck_triggered.evidence_sweep.sweep import run_phase


class _Dynamic:
    num_layers = 1
    disable_adaptation = False
    apply_ntk_reparam = False
    apply_experimental_rescaling = False
    activation_outer = staticmethod(F.leaky_relu)
    activation = staticmethod(F.relu6)

    def __init__(self, layers):
        self.layers = layers

    def get_layer(self, layer_id):
        return self.layers[layer_id]


class _TinyAWN(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, width), nn.Linear(width, 2)])
        self.dyn_model = _Dynamic(self.layers)
        self.qW_probs = [torch.linspace(1, 0.12, width)]

    def forward(self, x, skip_update=True):
        hidden = F.leaky_relu(F.relu6(self.layers[0](x)) * self.qW_probs[0])
        return (self.layers[1](hidden),)


def test_all_996_candidates_using_three_measured_checkpoints(monkeypatch, tmp_path):
    torch.manual_seed(8)
    data = torch.randn(48, 2)
    labels = (data[:, 0] > 0).long()
    calibration = DataLoader(TensorDataset(data[:32], labels[:32]), batch_size=8)
    validation = DataLoader(TensorDataset(data[32:], labels[32:]), batch_size=8)
    model = _TinyAWN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    monkeypatch.setattr(
        export_awn, "_static_graph_scores",
        lambda w_in, w_out, threshold: np.arange(len(w_in), dtype=float),
    )
    history = []
    ids = [f"toy:n:{i}" for i in range(8)]
    for step in range(1, 4):
        model.train()
        for x, y in calibration:
            optimizer.zero_grad()
            F.cross_entropy(model(x)[0], y).backward()
            optimizer.step()
        ev = tmp_path / f"ev{step}.npz"
        val = tmp_path / f"val{step}.npz"
        export_awn.export_checkpoint(
            model, calibration, validation, ids, ev, val,
            checkpoint_id=f"toy:epoch{step}",
            prior_evidence_paths=history[-2:] if step == 3 else (),
        )
        history.append(ev)
    with np.load(history[-1], allow_pickle=False) as x:
        assert "score_H" in x.files
    first = run_phase(1, history[-1], tmp_path / "stage1.json", val)
    second = run_phase(2, history[-1], tmp_path / "stage2.json", val)
    assert (first["ranked"], second["ranked"]) == (255, 741)
    assert (first["skipped"], second["skipped"]) == (0, 0)
    assert set(first["features_available"]) == set("WSAGJEHI")
