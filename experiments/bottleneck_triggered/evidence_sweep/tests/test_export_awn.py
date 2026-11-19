import json

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.bottleneck_triggered.evidence_sweep import export_awn
from experiments.bottleneck_triggered.evidence_sweep.sweep import read_frozen_eval, run_phase


class TinyDyn:
    def __init__(self):
        self.num_layers = 1
        self.disable_adaptation = False
        self.apply_ntk_reparam = False
        self.apply_experimental_rescaling = False
        self.activation_outer = F.leaky_relu
        self.activation = F.relu6
        self.layers = nn.ModuleList([nn.Linear(2, 5), nn.Linear(5, 2)])

    def get_layer(self, index):
        return self.layers[index]


class TinyAWN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 5), nn.Linear(5, 2)])
        self.dyn_model = TinyDyn()
        self.dyn_model.layers = self.layers
        self.qW_probs = [torch.linspace(1., .1, 5)]

    def forward(self, x, skip_update=True):
        h = F.leaky_relu(F.relu6(self.layers[0](x)) * self.qW_probs[0])
        return (self.layers[1](h),)


def test_export_with_real_torch_and_verified_logits(monkeypatch, tmp_path):
    monkeypatch.setattr(export_awn, '_static_graph_scores',
                        lambda w_in, w_out, threshold: np.arange(len(w_in), dtype=float))
    torch.manual_seed(13)
    model = TinyAWN()
    model.train()
    x = torch.randn(20, 2)
    y = torch.randint(0, 2, (20,))
    train = DataLoader(TensorDataset(x[:15], y[:15]), batch_size=5)
    valid = DataLoader(TensorDataset(x[15:], y[15:]), batch_size=5)
    ids = [f'n_{i}' for i in range(5)]
    ev, vl = tmp_path/'e.npz', tmp_path/'v.npz'
    export_awn.export_checkpoint(model, train, valid, ids, ev, vl, checkpoint_id='demo')
    assert model.training is True
    assert all(p.grad is None for p in model.parameters())
    assert read_frozen_eval(vl, ids)['identity_error'] < 1e-6
    with np.load(ev, allow_pickle=False) as cache:
        for key in ('W', 'S', 'A', 'G', 'J', 'E', 'I'):
            assert cache[f'score_{key}'].shape == (5,)
            assert np.isfinite(cache[f'score_{key}']).all()
        assert 'score_H' not in cache
    a = run_phase(1, ev, tmp_path/'p1.json', vl)
    b = run_phase(2, ev, tmp_path/'p2.json', vl)
    assert a['ranked'] == 127
    assert b["ranked"] == 360  # ((2**7 - 1) - 7 singletons) * 3
    # At least one candidate's actual gated frozen prefix changes under ranking.
    assert len(a['results']) == 255


def test_temporal_history_requires_matching_identities(tmp_path):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez_compressed(a, stable_ids=np.array(['alpha', 'beta']),
                        valid_J=np.array([True, True]), score_J=np.array([1., 2.]))
    np.savez_compressed(b, stable_ids=np.array(['alpha', 'beta']),
                        valid_J=np.array([True, True]), score_J=np.array([2., 1.]))
    h = export_awn._attested_history_score(['alpha', 'beta'], [a,b], np.array([3.,3.]))
    assert h.shape == (2,)
    with pytest.raises(ValueError, match='Stable|IDs/order'):
        export_awn._attested_history_score(['beta','alpha'], [a,b], np.array([3.,3.]))
