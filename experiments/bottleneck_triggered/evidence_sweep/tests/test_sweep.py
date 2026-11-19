import json
from pathlib import Path

import numpy as np
import pytest

from experiments.bottleneck_triggered.evidence_sweep.sweep import (
    candidate_configs, fuse, percentiles, read_frozen_eval, run_phase,
)


def create_fixture(tmp_path, *, omit=(), n=6, with_eval=True):
    rng = np.random.default_rng(42)
    stable = np.array([f"l0.neuron{i:03d}" for i in range(n)])
    features = dict(stable_ids=stable, physical_slots=np.arange(n),
                    active_mask=np.ones(n, dtype=bool),
                    metadata_json=json.dumps({"granularity": "neuron", "stage": "frozen_checkpoint", "fixture": True}))
    for idx, family in enumerate("WSAGJEHI"):
        if family not in omit:
            features[f"score_{family}"] = rng.normal(loc=idx, scale=1.0, size=n)
            features[f"valid_{family}"] = np.ones(n, dtype=bool)
    evid = tmp_path / "fixture_evidence.npz"
    np.savez_compressed(evid, **features)
    eval_path = None
    if with_eval:
        h = rng.normal(size=(12, n))
        w = rng.normal(size=(3, n))
        b = rng.normal(size=3)
        gates = np.geomspace(1.0, 0.1, n)
        labels = rng.integers(0, 3, size=len(h))
        x = h * gates
        x = np.where(x >= 0.0, x, 0.01 * x)
        logits = x @ w.T + b
        eval_path = tmp_path / "fixture_eval.npz"
        np.savez_compressed(eval_path, stable_ids=stable, hidden_pre_gate=h,
                            outgoing_weight=w, output_bias=b, position_gates=gates,
                            labels=labels, reference_logits=logits,
                            metadata_json=json.dumps({"evaluation_role": "validation", "activation_outer": "leaky_relu"}))
    return evid, eval_path


def test_manifest_counts_and_independence():
    one = candidate_configs(1)
    two = candidate_configs(2)
    assert (len(one), len(two)) == (255, 741)
    assert len(set(one + two)) == 996
    assert not set(one) & set(two)
    assert all(f == "mean_percentile" for _, f in one)
    assert all(len(s) > 1 and f != "mean_percentile" for s, f in two)


def test_run_stages_both_orders(tmp_path):
    evid, eval_path = create_fixture(tmp_path)
    for phase, size in ((2, 741), (1, 255)):
        path = tmp_path / f"stage{phase}.json"
        summary = run_phase(phase, evid, path, eval_path)
        assert summary["planned"] == summary["ranked"] == summary["evaluated"] == size
        assert summary["skipped"] == 0
        assert len(summary["results"]) == size
        assert set(summary["controls"]) == {"identity", "random", "reverse_physical"}
        assert summary["distinct_rankings"] >= 1
        assert all("prefix_auc_delta_vs_identity" in row for row in summary["results"])
        assert all(len(set(row["stable_id_order"])) == 6 for row in summary["results"])
        assert all(0 <= row["frozen_checkpoint_evaluation"]["prefix_accuracy_auc"] <= 0.9
                   for row in summary["results"])
        assert json.loads(path.read_text())["phase"] == phase
    assert run_phase(1, evid, tmp_path / "repeat.json", eval_path)["results"] == (
        json.loads((tmp_path / "stage1.json").read_text())["results"])


def test_partial_caches_are_audited_not_imputed(tmp_path):
    evid, _ = create_fixture(tmp_path, omit=("E", "H"), with_eval=False)
    report = run_phase(1, evid, tmp_path / "skips.json")
    assert report["planned"] == 255
    assert report["ranked"] == 2**6 - 1 == 63
    assert report["skipped"] == 192
    assert any(row["status"] == "skipped_unobserved" for row in report["results"])


def test_tied_scores_and_empty_gradients():
    assert np.all(percentiles(np.ones(5)) == 0.5)
    p = percentiles(np.array([1., 1., 2., 3.]))
    assert p[0] == p[1]
    assert p[2] > p[1] and p[3] > p[2]
    scores = {"W": np.ones(3), "S": np.array([0.0, 1.0, 2.0])}
    assert fuse(("W", "S"), "mean_percentile", scores)[2] > fuse(("W", "S"), "mean_percentile", scores)[0]


def test_pareto_dominance():
    vals = {"W": np.array([4., 2., 1.]), "G": np.array([4., 3., 0.])}
    scores = fuse(("W", "G"), "pareto_fronts", vals)
    assert scores[0] > scores[1] > scores[2]


def test_no_test_leakage(tmp_path):
    _, ev = create_fixture(tmp_path)
    with np.load(ev, allow_pickle=False) as arr:
        payload = {k: arr[k] for k in arr.files}
    payload["metadata_json"] = json.dumps({"evaluation_role": "test", "activation_outer": "leaky_relu"})
    np.savez_compressed(ev, **payload)
    with pytest.raises(ValueError, match="test split"):
        read_frozen_eval(ev, [f"l0.neuron{i:03d}" for i in range(6)])


def test_reference_mismatch_detected(tmp_path):
    _, ev = create_fixture(tmp_path)
    with np.load(ev, allow_pickle=False) as arr:
        payload = {k: arr[k] for k in arr.files}
    payload["reference_logits"] = payload["reference_logits"] + 0.1
    np.savez_compressed(ev, **payload)
    with pytest.raises(ValueError, match="reconstruction failed"):
        read_frozen_eval(ev, [f"l0.neuron{i:03d}" for i in range(6)])


def test_missing_gradient_is_not_zero(tmp_path):
    evid, _ = create_fixture(tmp_path, with_eval=False)
    with np.load(evid, allow_pickle=False) as arr:
        payload = {k: arr[k] for k in arr.files}
    payload["valid_G"][2] = False
    np.savez_compressed(evid, **payload)
    report = run_phase(2, evid, tmp_path / "badgrad.json")
    assert report["skipped"] > 0
    assert "G" in report["feature_issues"]
    assert report["ranked"] < 741


def test_stable_identity_ties(tmp_path):
    evid, _ = create_fixture(tmp_path, with_eval=False)
    with np.load(evid, allow_pickle=False) as arr:
        payload = {k: arr[k] for k in arr.files}
    payload["score_W"][:] = 1.0
    ids = payload["stable_ids"].copy()
    payload["stable_ids"] = ids[::-1]
    np.savez_compressed(evid, **payload)
    report = run_phase(1, evid, tmp_path / "ties.json")
    row = next(x for x in report["results"] if x["id"] == "EW__mean_percentile")
    assert row["stable_id_order"] == sorted(ids.tolist())
