"""Fail-closed aggregation and seed-clustered descriptive summaries of 96 forks.

No p-values, cross-task transfer claims, or tuned policy are computed here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from .run_grid import CHECKPOINTS, DATASETS, HORIZON_STEPS, SEEDS

ACTIONS = ("skip", "random_positional", "weight_positional", "gradient_positional")


def aggregate_reports(reports: list[dict]) -> dict:
    expected = {(dataset, seed) for dataset in DATASETS for seed in SEEDS}
    if len(reports) != len(expected):
        raise AssertionError("Need precisely six complete dataset/seed groups")
    seen = {}
    commits = set()
    comparisons = []
    checkpoint_ids = set()
    for r in reports:
        key = (r["dataset"], r["seed"])
        if key not in expected or key in seen or not r["passed"]:
            raise AssertionError("Missing/duplicate/invalid dataset-seed report")
        if r["test_examples_used"] != 0 or r["checkpoint_epochs"] != list(CHECKPOINTS):
            raise AssertionError("Leaked test data or altered checkpoint design")
        if (r["final_horizon_rollouts"] != 16
                or r["all_horizon_observations"] != 48
                or len(r["checkpoints"]) != 4):
            raise AssertionError("Incorrect checkpoint treatment budget")
        if r["split"]["test_examples_used"] != 0:
            raise AssertionError("Audit reports test leakage")
        seen[key] = r
        commits.add(r["git_sha"])
        for cp in r["checkpoints"]:
            if cp["epoch"] not in CHECKPOINTS or cp["checkpoint_id"] in checkpoint_ids:
                raise AssertionError("Duplicated or undeclared checkpoint")
            checkpoint_ids.add(cp["checkpoint_id"])
            rows = cp["final_horizon_comparison"]
            if len(rows) != len(ACTIONS) or {x["action"] for x in rows} != set(ACTIONS):
                raise AssertionError("Action grid incomplete")
            for row in rows:
                if row["gate_mode"] != "positional" or row["optimizer_steps"] != HORIZON_STEPS:
                    raise AssertionError("Wrong action semantics or horizon")
                if row["training_examples"] != rows[0]["training_examples"]:
                    raise AssertionError("Unequal training examples")
                if not math.isfinite(row["delta_risk"]) or not math.isfinite(row["net_value"]):
                    raise AssertionError("Missing or nonfinite value labels")
                comparisons.append({
                    "dataset": r["dataset"], "seed": r["seed"],
                    "checkpoint_epoch": cp["epoch"],
                    "checkpoint_id": cp["checkpoint_id"],
                    **row,
                })
    if len(commits) != 1:
        raise AssertionError("Do not aggregate results from different source commits")
    if seen.keys() != expected or len(comparisons) != 96:
        raise AssertionError("Incomplete 96-rollout action grid")
    summaries = []
    for dataset in DATASETS:
        for action in ACTIONS:
            rows = [x for x in comparisons if x["dataset"] == dataset
                    and x["action"] == action]
            if len(rows) != 12:
                raise AssertionError("Missing dataset-action comparisons")
            seed_means = {
                str(s): statistics.mean(x["delta_risk"] for x in rows if x["seed"] == s)
                for s in SEEDS
            }
            seed_net_means = {
                str(s): statistics.mean(x["net_value"] for x in rows if x["seed"] == s)
                for s in SEEDS
            }
            effective = [x for x in rows if not x["no_op"]]
            summaries.append({
                "dataset": dataset, "action": action,
                "checkpoints": len(rows), "seeds": len(SEEDS),
                "effective_interventions": len(effective),
                "mean_delta_risk": statistics.mean(x["delta_risk"] for x in rows),
                "median_delta_risk": statistics.median(x["delta_risk"] for x in rows),
                "improved_risk_count": sum(
                    x["delta_risk"] > 0 for x in effective),
                "mean_net_value": statistics.mean(x["net_value"] for x in rows),
                "seed_mean_delta_risk": seed_means,
                "seed_mean_net_value": seed_net_means,
                "seed_cluster_sd_delta_risk": statistics.stdev(
                    seed_means.values()),
            })
    return {
        "schema": "awn_structural_value_grid96_aggregate_v1",
        "git_sha": next(iter(commits)),
        "datasets": list(DATASETS),
        "seeds": list(SEEDS), "checkpoints": list(CHECKPOINTS),
        "actions": list(ACTIONS), "horizon_optimizer_steps": HORIZON_STEPS,
        "dataset_seed_runs": len(seen), "checkpoint_decisions": len(checkpoint_ids),
        "final_horizon_rollouts": len(comparisons),
        "all_horizon_observations": sum(
            r["all_horizon_observations"] for r in reports),
        "final_treatment_comparisons": comparisons,
        "descriptive_summaries": summaries,
        "test_examples_used": 0,
        "passed": True,
        "interpretation": "Engineering sample; three seeds not confirmatory inference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for dataset in DATASETS:
        for seed in SEEDS:
            name = args.root / f"grid_{dataset}_seed{seed}" / "grid_audit.json"
            if not name.is_file():
                raise FileNotFoundError(f"Missing mandatory grid audit: {name}")
            reports.append(json.loads(name.read_text(encoding="utf-8")))
    result = aggregate_reports(reports)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a completed aggregation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "passed": result["passed"],
        "checkpoint_decisions": result["checkpoint_decisions"],
        "final_horizon_rollouts": result["final_horizon_rollouts"],
        "all_horizon_observations": result["all_horizon_observations"],
    }, indent=2))


if __name__ == "__main__":
    main()
