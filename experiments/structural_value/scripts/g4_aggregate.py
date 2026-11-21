"""Stage C: read only complete, independently audited G4 outcome artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from ..g4_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, ACTIONS, HORIZON_STEPS,
    COST_WEIGHT, REFERENCE_SECONDS, G3_SOURCE_RUN, G3_SOURCE_ARTIFACT_ID,
)
from ..g4_frozen_policy import _json_sha
from .audit_pilot import audit_pilot

POLICIES = ("locked_ridge", "always_skip", "always_random",
            "always_w", "always_g", "hindsight_oracle_diagnostic")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssertionError("Non-numeric prospective confirmation value")
    number = float(value)
    if not math.isfinite(number):
        raise AssertionError("Nonfinite prospective confirmation value")
    return number


def collect(root: Path) -> tuple[list[dict], dict]:
    expected = {(ds, seed) for ds in DATASETS for seed in SEEDS}
    decisions = []
    summary_groups = []
    provenance_sha = set()
    policy_sha = set()
    checkpoint_ids = set()
    for ds, seed in sorted(expected):
        folder = Path(root) / f"g4_{ds}_seed{seed}"
        report = json.loads((folder / "confirmation.json").read_text(encoding="utf-8"))
        precommit_file = folder / "precommit_original.json"
        pre = json.loads(precommit_file.read_text(encoding="utf-8"))
        design = pre["design"]
        if (report.get("schema") != "awn_structural_value_g4_confirmation_v1"
                or not report["passed"] or report["dataset"] != ds
                or report["seed"] != seed or not len(report["checkpoint_results"]) == 4
                or report["test_examples_used"] != 0
                or report["source_precommit_sha256"] != _sha_file(precommit_file)
                or pre["schema"] != "awn_structural_value_g4_precommitted_predictions_v1"
                or pre["outcome_files_created"] != 0
                or pre["test_examples_used"] != 0
                or design["dataset"] != ds or design["seed"] != seed
                or design["checkpoint_epochs"] != list(CHECKPOINTS)
                or design["rollout_horizons"] != list(HORIZONS)
                or design["horizon_steps"] != HORIZON_STEPS
                or design["actions"] != list(ACTIONS)
                or design["cost_weight"] != COST_WEIGHT
                or design["reference_seconds"] != REFERENCE_SECONDS
                or design["source_g3_run"] != G3_SOURCE_RUN
                or design["source_g3_artifact_id"] != G3_SOURCE_ARTIFACT_ID
                or report["source_g3_run"] != G3_SOURCE_RUN
                or report["source_g3_artifact_id"] != G3_SOURCE_ARTIFACT_ID
                or report["frozen_model_sha256"] != design["frozen_model_sha256"]
                or _json_sha(pre["locked_model"]) != design["frozen_model_sha256"]
                or design["source_commit"] != report["source_commit"]
                or report["run_id"] != pre["run_id"]
                or pre["split"] != report["split"]
                or report["split"]["test_examples_used"] != 0
                or report["final_horizon_rollouts"] != 16
                or report["all_horizon_observations"] != 48):
            raise AssertionError("Source-locked prospective study group invalid")
        provenance_sha.add(report["source_commit"])
        policy_sha.add(report["frozen_model_sha256"])
        if [r["epoch"] for r in report["checkpoint_results"]] != list(CHECKPOINTS):
            raise AssertionError("Checkpoint design or order changed")
        if [r["epoch"] for r in pre["checkpoint_decisions"]] != list(CHECKPOINTS):
            raise AssertionError("Prediction stage checkpoint design changed")
        for result, decision in zip(report["checkpoint_results"], pre["checkpoint_decisions"]):
            epoch = decision["epoch"]
            if (result["checkpoint_id"] != decision["checkpoint_id"]
                    or result["precommitted_action"] != decision["predicted_action"]
                    or result["precommitted_net_value_predictions"] != decision["predicted_net_value"]
                    or result["precommitted_state"] != decision["state"]
                    or result["snapshot_sha256"] != decision["snapshot_sha256"]
                    or not result["audit_passed"]
                    or result["intermediate_observations"] != 12):
                raise AssertionError("Post-outcome decision differs from prior predictions")
            if result["checkpoint_id"] in checkpoint_ids:
                raise AssertionError("Same checkpoint reused in separate study groups")
            checkpoint_ids.add(result["checkpoint_id"])
            cp_dir = folder / f"epoch{epoch:04d}"
            manifest = json.loads((cp_dir / "manifest.json").read_text(encoding="utf-8"))
            if (manifest["checkpoint_id"] != result["checkpoint_id"]
                    or manifest["precommitted_action"] != result["precommitted_action"]
                    or manifest["precommit_model_sha256"] != report["frozen_model_sha256"]
                    or manifest["decision_state"] != decision["state"]
                    or manifest["config"]["git_sha"] != report["source_commit"]):
                raise AssertionError("Raw observed checkpoint or locked action mismatched")
            refreshed = audit_pilot(cp_dir)
            stored = json.loads((cp_dir / "audit.json").read_text(encoding="utf-8"))
            if (refreshed != stored or not refreshed["passed"]
                    or refreshed["actions"] != len(ACTIONS)
                    or refreshed["paired_horizons"] != list(HORIZONS)
                    or refreshed["observations"] != 12 or refreshed["labels"] != 12):
                raise AssertionError("Tampered or incomplete prospective matched observations")
            raw = refreshed["final_horizon_comparison"]
            by_action = {r["action"]: r for r in raw}
            if len(by_action) != len(ACTIONS) or set(by_action) != set(ACTIONS):
                raise AssertionError("Incomplete actual treatment and skip branches")
            if raw != result["all_observed_actions_at_h8"]:
                raise AssertionError("Reported results differ from source outcome")
            chosen = result["precommitted_action"]
            if chosen not in ACTIONS:
                raise AssertionError("Unapproved prospectively selected action")
            if (abs(_finite(result["selected_observed_net_value"])
                    - _finite(by_action[chosen]["net_value"])) > 1e-10
                    or abs(_finite(result["selected_observed_delta_risk"])
                    - _finite(by_action[chosen]["delta_risk"])) > 1e-10):
                raise AssertionError("Policy selected outcome was retrospectively altered")
            if result["selected_no_op"] != by_action[chosen]["no_op"]:
                raise AssertionError("Policy no-op status corrupted")
            oracle = max(ACTIONS, key=lambda a: by_action[a]["net_value"])
            choices = {
                "locked_ridge": chosen,
                "always_skip": "skip",
                "always_random": "random_positional",
                "always_w": "weight_positional",
                "always_g": "gradient_positional",
                "hindsight_oracle_diagnostic": oracle,
            }
            row = {
                "dataset": ds, "seed": seed, "epoch": epoch,
                "checkpoint_id": result["checkpoint_id"],
                "precommitted_action": chosen,
                "predicted_values": decision["predicted_net_value"],
                "observed_action_values": {
                    a: _finite(by_action[a]["net_value"]) for a in ACTIONS
                },
                "policy_realized_values": {
                    name: _finite(by_action[action]["net_value"])
                    for name, action in choices.items()
                },
                "policy_risk_differences": {
                    name: _finite(by_action[action]["delta_risk"])
                    for name, action in choices.items()
                },
                "policy_time_differences": {
                    name: _finite(by_action[action]["delta_online_seconds"])
                    for name, action in choices.items()
                },
                "selected_no_op": bool(result["selected_no_op"]),
                "selected_action": chosen,
            }
            decisions.append(row)
        summary_groups.append({
            "dataset": ds, "seed": seed, "checkpoints": 4,
            "rollouts": 16, "observations": 48, "passed": True,
        })
    if len(provenance_sha) != 1 or len(policy_sha) != 1:
        raise AssertionError("Mixed source commits or predictor models in confirmation")
    if len(decisions) != 24 or len(checkpoint_ids) != 24:
        raise AssertionError("G4 missing or duplicated prospectively decided state")
    return decisions, {
        "source_commit": next(iter(provenance_sha)),
        "frozen_model_sha256": next(iter(policy_sha)),
        "groups": summary_groups,
    }


def summarize(root: Path) -> dict:
    decisions, provenance = collect(root)
    def score(rows):
        return {
            policy: {
                "mean_net_value": statistics.mean(
                    x["policy_realized_values"][policy] for x in rows),
                "median_net_value": statistics.median(
                    x["policy_realized_values"][policy] for x in rows),
                "mean_risk_improvement": statistics.mean(
                    x["policy_risk_differences"][policy] for x in rows),
                "beneficial": sum(x["policy_realized_values"][policy] > 0 for x in rows),
                "harmful": sum(x["policy_realized_values"][policy] < 0 for x in rows),
                "worst_net_value": min(x["policy_realized_values"][policy] for x in rows),
            } for policy in POLICIES
        }
    cohort = {}
    for dataset in DATASETS:
        subset = [d for d in decisions if d["dataset"] == dataset]
        cohort[dataset] = score(subset)
    seed_means = {
        str(seed): statistics.mean(
            d["policy_realized_values"]["locked_ridge"]
            for d in decisions if d["seed"] == seed)
        for seed in SEEDS
    }
    return {
        "schema": "awn_structural_value_g4_untouched_seed_confirmation_v1",
        "run_design": "new seeds precommitted before paired outcomes",
        "source_g3_run": G3_SOURCE_RUN,
        "source_g3_artifact_id": G3_SOURCE_ARTIFACT_ID,
        **provenance,
        "datasets": list(DATASETS), "seeds": list(SEEDS),
        "checkpoint_epochs": list(CHECKPOINTS),
        "confirmation_decisions": len(decisions),
        "final_horizon_rollouts": len(decisions) * len(ACTIONS),
        "all_horizon_observations": len(decisions) * len(ACTIONS) * len(HORIZONS),
        "action_counts": {
            a: sum(d["selected_action"] == a for d in decisions) for a in ACTIONS
        },
        "policy_scores": score(decisions),
        "by_dataset": cohort,
        "heldout_seed_mean_locked_policy": seed_means,
        "selected_no_op_count": sum(d["selected_no_op"] for d in decisions),
        "individual_decisions": decisions,
        "test_examples_used": 0, "passed": True,
        "interpretation": (
            "Fresh-seed precommitted offline native-AWN confirmation only; "
            "limited to two reference datasets and short-horizon cost metric"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a prior G4 confirmation")
    outcome = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        outcome, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": outcome["passed"],
        "confirmation_decisions": outcome["confirmation_decisions"],
        "rollouts": outcome["final_horizon_rollouts"],
        "observations": outcome["all_horizon_observations"],
        "action_counts": outcome["action_counts"],
        "policies": outcome["policy_scores"],
        "by_dataset": outcome["by_dataset"],
        "per_seed_ridge_means": outcome["heldout_seed_mean_locked_policy"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
