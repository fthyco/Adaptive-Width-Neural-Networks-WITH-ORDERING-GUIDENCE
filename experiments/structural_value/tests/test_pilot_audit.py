"""Unit audits protect the real dataset pilot output contract."""
import hashlib
import json
from pathlib import Path
import pytest

from experiments.structural_value.scripts.audit_pilot import audit_pilot
from experiments.structural_value.actions.action_spec import ActionKind, ActionSpec


def _valid_dir(tmp_path: Path) -> Path:
    cfg = {"dataset": "Spiral", "seed": 42, "rollout_seed": 99,
           "cost_weight": 0.1, "reference_seconds": 1.0,
           "git_sha": "f" * 40}
    actions = [ActionSpec(ActionKind.SKIP), ActionSpec(ActionKind.WEIGHT_POS)]
    aid = [a.action_id for a in actions]
    manifest = {
        "schema": "structural_value_pilot_v1", "config": cfg,
        "config_hash": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
        "run_id": "r", "checkpoint_id": "r:epoch3",
        "split": {"train_examples": 60, "calibration_examples": 10,
                  "screening_validation_examples": 10, "test_examples_used": 0,
                  "provenance": "train_derived_validation"},
        "training_population": 60, "rollout_endpoints": [1],
        "rollout_optimizer_steps": 1, "materialized_batch_sha256": "a" * 64,
        "actions": [a.payload for a in actions],
        "paired_observations": 2, "paired_labels": 2,
        "test_examples_used_for_selection": 0,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows, labels = [], []
    for i, a in enumerate(actions):
        risk, cost = (0.5, 2.0) if i == 0 else (0.4, 2.1)
        row = {
            "run_id": "r", "checkpoint_id": "r:epoch3",
            "rollout_seed": 99, "horizon_steps": 1,
            "action_id": aid[i], "action": a.payload,
            "validation_risk": risk, "immediate_risk_before": 0.6,
            "immediate_risk_after": (0.6 if i == 0 else 0.55),
            "no_op": (i == 0), "final_widths": [10],
            "cost": {"optimizer_steps": 1, "examples": 6,
                     "evidence_seconds": 0.0 if i == 0 else 0.05,
                     "intervention_seconds": 0.0 if i == 0 else 0.05,
                     "training_seconds": 1.0, "evaluation_seconds": 1.0,
                     "policy_seconds": 0.0, "online_seconds": cost},
        }
        rows.append(row)
        labels.append({
            "run_id": "r", "checkpoint_id": "r:epoch3",
            "rollout_seed": 99, "horizon_steps": 1,
            "action_id": aid[i], "delta_risk": 0.0 if i == 0 else 0.1,
            "delta_online_seconds": 0.0 if i == 0 else 0.1,
            "net_value": 0.0 if i == 0 else 0.09,
            "no_op": (i == 0),
        })
    for name, content in (("observations.jsonl", rows), ("labels.jsonl", labels)):
        (tmp_path / name).write_text("".join(json.dumps(x) + "\n" for x in content))
    return tmp_path


def test_valid_pilot_passes_and_records_net_value(tmp_path):
    audit = audit_pilot(_valid_dir(tmp_path))
    assert audit["passed"]
    assert audit["final_horizon_comparison"][1]["net_value"] == pytest.approx(0.09)


@pytest.mark.parametrize("corruption", ["test_leak", "wrong_label", "unequal_steps",
                                       "duplicate_observation", "nonfinite_risk"])
def test_fail_closed_on_corrupt_results(tmp_path, corruption):
    folder = _valid_dir(tmp_path)
    def edit(name, fn):
        path = folder / name
        if name == "manifest.json":
            payload = json.loads(path.read_text())
            fn(payload)
            path.write_text(json.dumps(payload))
        else:
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            fn(rows)
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    if corruption == "test_leak":
        edit("manifest.json", lambda x: x["split"].update(test_examples_used=1))
    elif corruption == "wrong_label":
        edit("labels.jsonl", lambda x: x[1].update(delta_risk=-0.1))
    elif corruption == "unequal_steps":
        edit("observations.jsonl", lambda x: x[1]["cost"].update(examples=4))
    elif corruption == "duplicate_observation":
        edit("observations.jsonl", lambda x: x[1].update(action_id=x[0]["action_id"]))
    elif corruption == "nonfinite_risk":
        edit("observations.jsonl", lambda x: x[1].update(validation_risk=float("nan")))
    with pytest.raises((AssertionError, ValueError)):
        audit_pilot(folder)
