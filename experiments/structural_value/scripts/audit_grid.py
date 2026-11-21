"""Validate all checkpoints in a single dataset/seed grid without test leakage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit_pilot import audit_pilot, _finite
from ..grid_contract import CHECKPOINTS, DATASETS, HORIZON_STEPS, HORIZONS, SEEDS


def audit_grid(folder: Path) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / "grid_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "awn_structural_value_grid96_per_run_v1":
        raise AssertionError("Incorrect grid manifest schema")
    design = manifest["design"]
    dataset, seed = design["dataset"], design["seed"]
    if dataset not in DATASETS or seed not in SEEDS:
        raise AssertionError("Grid dataset or seed not predeclared")
    if (design["checkpoint_epochs"] != list(CHECKPOINTS)
            or design["horizons"] != list(HORIZONS)
            or design["max_rollout_steps"] != HORIZON_STEPS
            or design["actions"] != [
                "skip", "random_positional", "weight_positional",
                "gradient_positional"
            ]):
        raise AssertionError("Grid design deviated from predeclared contract")
    if design["cost_weight"] != 0.01 or design["reference_seconds"] != 1.0:
        raise AssertionError("Post-hoc change to cost exchange rate")
    if (manifest["test_examples_used_for_selection"] != 0
            or design["test_examples_for_labels"] != 0
            or manifest["split"]["test_examples_used"] != 0):
        raise AssertionError("Test examples were used to select actions")
    if not manifest["passed"] or len(manifest["checkpoints"]) != len(CHECKPOINTS):
        raise AssertionError("Missing model checkpoints")
    if [r["epoch"] for r in manifest["checkpoints"]] != list(CHECKPOINTS):
        raise AssertionError("Checkpoint epoch ordering was altered")
    history = manifest["training_history"]
    if len(history) != CHECKPOINTS[-1] or [
        r["epoch"] for r in history] != list(range(1, CHECKPOINTS[-1] + 1)):
        raise AssertionError("Model history did not come from continuous training")
    steps_so_far = 0
    for r in history:
        steps_so_far += r["training_steps"]
        if r["training_steps"] <= 0 or r["training_examples"] != manifest["training_population"]:
            raise AssertionError("Native AWN baseline training is incomplete")
        for field in ("mean_train_elbo_loss", "validation_ce",
                      "validation_accuracy"):
            _finite(r[field], field)
    observed = []
    total_final = 0
    total_observations = 0
    for entry in manifest["checkpoints"]:
        epoch = entry["epoch"]
        folder_name = f"epoch{epoch:04d}"
        if entry["pilot_subdir"] != folder_name:
            raise AssertionError("Checkpoint provenance path was altered")
        cp_dir = folder / folder_name
        pil = json.loads((cp_dir / "manifest.json").read_text(encoding="utf-8"))
        if (pil["run_id"] != manifest["run_id"]
                or pil["checkpoint_id"] != entry["checkpoint_id"]
                or pil["config"]["dataset"] != dataset
                or pil["config"]["seed"] != seed
                or pil["config"]["checkpoint_epoch"] != epoch
                or pil["config"]["git_sha"] != manifest["git_sha"]
                or pil["config"]["checkpoint_set"] != list(CHECKPOINTS)
                or pil["config"]["max_rollout_steps"] != HORIZON_STEPS
                or pil["rollout_endpoints"] != list(HORIZONS)
                or pil["split"] != manifest["split"]):
            raise AssertionError("Checkpoint provenance or design differs from grid")
        state = pil["decision_state"]
        hist = history[epoch - 1]
        if (state != entry["decision_state"]
                or state["epoch"] != epoch
                or state["physical_width"] != hist["physical_width"]
                or state["learned_width"] != hist["learned_width"]
                or state["validation_ce"] != hist["validation_ce"]
                or state["train_elbo_loss"] != hist["mean_train_elbo_loss"]
                or state["training_steps_so_far"] != sum(
                    r["training_steps"] for r in history[:epoch])):
            raise AssertionError("Decision features do not match prior-only AWN history")
        audit = audit_pilot(cp_dir)
        if (not audit["passed"] or audit["actions"] != 4
                or audit["paired_horizons"] != list(HORIZONS)
                or audit["observations"] != 12 or audit["labels"] != 12):
            raise AssertionError("Incomplete checkpoint action grid")
        if len(audit["final_horizon_comparison"]) != 4:
            raise AssertionError("A treatment final horizon is missing")
        if audit["checkpoint_id"] != entry["checkpoint_id"]:
            raise AssertionError("Incorrect audit checkpoint")
        if set((x["action"], x["gate_mode"]) for x in audit["final_horizon_comparison"]) != {
            ("skip", "positional"),
            ("random_positional", "positional"),
            ("weight_positional", "positional"),
            ("gradient_positional", "positional"),
        }:
            raise AssertionError("Unpredeclared action or missing skip")
        for row in audit["final_horizon_comparison"]:
            if row["optimizer_steps"] != HORIZON_STEPS:
                raise AssertionError("Treatment horizon mismatch")
            if row["no_op"] and row["action"] != "skip":
                # A no-op can be valid, but it cannot count as useful treatment.
                pass
        total_final += 4
        total_observations += audit["observations"]
        observed.append({
            "epoch": epoch, "checkpoint_id": entry["checkpoint_id"],
            "decision_state": state,
            "final_horizon_comparison": audit["final_horizon_comparison"],
            "observations": audit["observations"],
            "labels": audit["labels"], "passed": True,
        })
    if (total_final != 16 or total_observations != 48
            or manifest["total_final_horizon_rollouts"] != 16
            or manifest["total_horizon_observations"] != 48):
        raise AssertionError("Wrong number of independent treatment forks")
    return {
        "schema": "awn_structural_value_grid96_per_run_audit_v1",
        "dataset": dataset, "seed": seed,
        "git_sha": manifest["git_sha"], "run_id": manifest["run_id"],
        "split": manifest["split"],
        "checkpoint_epochs": list(CHECKPOINTS),
        "checkpoints": observed,
        "final_horizon_rollouts": total_final,
        "all_horizon_observations": total_observations,
        "test_examples_used": 0, "passed": True,
        "interpretation": "Multi-checkpoint engineering labels, not confirmation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_grid(args.input)
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing audit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "dataset": result["dataset"], "seed": result["seed"],
        "passed": result["passed"],
        "final_horizon_rollouts": result["final_horizon_rollouts"],
        "all_horizon_observations": result["all_horizon_observations"],
    }, indent=2))


if __name__ == "__main__":
    main()
