"""Fail-closed audit of a real-data AWN counterfactual pilot folder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise AssertionError(f"Missing pilot rows: {path}")
    return [json.loads(x) for x in lines]


def _finite(number, name) -> float:
    if isinstance(number, bool) or not isinstance(number, (float, int)):
        raise AssertionError(f"Missing numeric {name}")
    number = float(number)
    if not math.isfinite(number):
        raise AssertionError(f"Non-finite numeric {name}")
    return number


def audit_pilot(folder: Path) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    observations = _rows(folder / "observations.jsonl")
    labels = _rows(folder / "labels.jsonl")
    if manifest["schema"] != "structural_value_pilot_v1":
        raise AssertionError("Unexpected result schema")
    if manifest["config"]["dataset"] not in ("Spiral", "SpiralHard", "DoubleMoon"):
        raise AssertionError("Not a declared AWN benchmark dataset")
    if manifest["test_examples_used_for_selection"] != 0:
        raise AssertionError("Final test set leaked into selection")
    split = manifest["split"]
    if split["test_examples_used"] != 0:
        raise AssertionError("Test set accessed while fitting or selecting")
    if min(split["train_examples"], split["calibration_examples"],
           split["screening_validation_examples"]) <= 0:
        raise AssertionError("Nonpositive split sample count")
    if split["provenance"] not in ("provided_independent_validation",
                                   "train_derived_validation"):
        raise AssertionError("Unknown disjoint development split provenance")
    if manifest["training_population"] != split["train_examples"]:
        raise AssertionError("Incorrect AWN training denominator")
    stream_digest = manifest["materialized_batch_sha256"]
    if (not isinstance(stream_digest, str) or len(stream_digest) != 64
            or any(ch not in "0123456789abcdef" for ch in stream_digest)):
        raise AssertionError("Missing or malformed batch-stream fingerprint")
    config_digest = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()
    if manifest["config_hash"] != config_digest:
        raise AssertionError("Config metadata mismatch")

    endpoints = manifest["rollout_endpoints"]
    steps_total = manifest["rollout_optimizer_steps"]
    if sorted(set(endpoints)) != endpoints or not endpoints or endpoints[-1] != steps_total:
        raise AssertionError("Invalid observed optimizer horizons")
    expected_actions = {hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()[:20]
                        for a in manifest["actions"]}
    if len(expected_actions) != len(manifest["actions"]):
        raise AssertionError("Duplicate treatment identity")
    if len(observations) != len(labels) or len(observations) != len(endpoints) * len(expected_actions):
        raise AssertionError("Missing or duplicated paired observations/labels")
    if len(observations) != manifest["paired_observations"] or len(labels) != manifest["paired_labels"]:
        raise AssertionError("Manifest observation count mismatch")

    by_pair: dict[tuple, dict] = {}
    by_row: dict[tuple, dict] = {}
    action_summaries = {}
    for row in observations:
        if row["run_id"] != manifest["run_id"] or row["checkpoint_id"] != manifest["checkpoint_id"]:
            raise AssertionError("Checkpoint identity mismatch")
        if row["rollout_seed"] != manifest["config"]["rollout_seed"]:
            raise AssertionError("Rollout seed mismatch")
        if row["action_id"] not in expected_actions or row["horizon_steps"] not in endpoints:
            raise AssertionError("Unknown treatment or horizon")
        pair = (row["run_id"], row["checkpoint_id"], row["rollout_seed"], row["horizon_steps"])
        key = (pair, row["action_id"])
        if key in by_row:
            raise AssertionError("Duplicate endpoint/treatment row")
        by_row[key] = row
        _finite(row["validation_risk"], "validation_risk")
        _finite(row["immediate_risk_before"], "immediate_risk_before")
        _finite(row["immediate_risk_after"], "immediate_risk_after")
        cost = row["cost"]
        for k in ("evidence_seconds", "intervention_seconds", "training_seconds",
                  "evaluation_seconds", "policy_seconds", "online_seconds"):
            if _finite(cost[k], k) < 0:
                raise AssertionError(f"Negative cost component {k}")
        if cost["optimizer_steps"] != row["horizon_steps"] or cost["examples"] < row["horizon_steps"]:
            raise AssertionError("Invalid step/example accounting")
        if abs(cost["online_seconds"] - sum(cost[k] for k in (
            "evidence_seconds", "intervention_seconds", "training_seconds",
            "evaluation_seconds", "policy_seconds"))) > 1e-8:
            raise AssertionError("Cost components do not sum to online cost")
        if any(not isinstance(w, int) or w < 1 for w in row["final_widths"]):
            raise AssertionError("Invalid final AWN widths")
        by_pair.setdefault(pair, {})[row["action_id"]] = row
        action_summaries[row["action_id"]] = row["action"]

    for pair, group in by_pair.items():
        if set(group) != expected_actions:
            raise AssertionError("Incomplete matched action group")
        skips = [r for r in group.values() if r["action"]["kind"] == "skip"]
        if len(skips) != 1 or not skips[0]["no_op"]:
            raise AssertionError("Missing real AWN skip baseline")
        skip = skips[0]
        if skip["cost"]["evidence_seconds"] != 0 or skip["cost"]["intervention_seconds"] != 0:
            raise AssertionError("Skip may not carry evidence or intervention work")
        if skip["immediate_risk_before"] != skip["immediate_risk_after"]:
            raise AssertionError("Skip changed the immediate validation risk")
        for row in group.values():
            if (row["cost"]["examples"] != skip["cost"]["examples"] or
                    row["cost"]["optimizer_steps"] != skip["cost"]["optimizer_steps"]):
                raise AssertionError("Treatment has a different training budget")
    lookup = {}
    cost_weight = manifest["config"]["cost_weight"]
    ref_seconds = manifest["config"]["reference_seconds"]
    for label in labels:
        pair = (label["run_id"], label["checkpoint_id"], label["rollout_seed"],
                label["horizon_steps"])
        key = (pair, label["action_id"])
        if key in lookup or key not in by_row:
            raise AssertionError("Orphaned or duplicated label")
        lookup[key] = label
        row = by_row[key]
        skip = next(r for r in by_pair[pair].values() if r["action"]["kind"] == "skip")
        expected_risk = 0.0 if row["action"]["kind"] == "skip" else (
            skip["validation_risk"] - row["validation_risk"])
        expected_cost = 0.0 if row["action"]["kind"] == "skip" else (
            row["cost"]["online_seconds"] - skip["cost"]["online_seconds"])
        net = expected_risk - cost_weight * expected_cost / ref_seconds
        for k, expected in (("delta_risk", expected_risk),
                            ("delta_online_seconds", expected_cost),
                            ("net_value", net)):
            if abs(_finite(label[k], k) - expected) > 1e-9:
                raise AssertionError(f"Incorrect pairing/sign/normalization: {k}")
        if label["no_op"] != row["no_op"]:
            raise AssertionError("Label no-op flag differs from intervention")
    if len(lookup) != len(by_row):
        raise AssertionError("Incomplete matched teacher labels")
    final_rows = [row for row in observations if row["horizon_steps"] == steps_total]
    final_labels = [label for label in labels if label["horizon_steps"] == steps_total]
    comparison = []
    for row in final_rows:
        label = next(x for x in final_labels if x["action_id"] == row["action_id"])
        comparison.append({
            "action": row["action"]["kind"], "gate_mode": row["action"]["gate_mode"],
            "no_op": row["no_op"],
            "validation_ce": row["validation_risk"], "delta_risk": label["delta_risk"],
            "delta_online_seconds": label["delta_online_seconds"],
            "net_value": label["net_value"], "final_widths": row["final_widths"],
            "optimizer_steps": row["cost"]["optimizer_steps"],
            "training_examples": row["cost"]["examples"],
        })
    return {
        "schema": "awn_structural_value_dataset_audit_v1",
        "dataset": manifest["config"]["dataset"],
        "seed": manifest["config"]["seed"],
        "run_id": manifest["run_id"],
        "checkpoint_id": manifest["checkpoint_id"],
        "git_sha": manifest["config"]["git_sha"],
        "split": split,
        "paired_horizons": endpoints,
        "actions": len(expected_actions),
        "observations": len(observations),
        "labels": len(labels),
        "training_budget_equal": True,
        "test_examples_used": 0,
        "label_equations_verified": True,
        "final_horizon_comparison": comparison,
        "passed": True,
        "interpretation": "Engineering pilot only; no confirmatory treatment ranking",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_pilot(args.input)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError("Audit report cannot overwrite an existing file")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
