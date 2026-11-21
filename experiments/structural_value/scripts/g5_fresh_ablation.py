"""Re-audit fresh same-checkpoint G5A study and compare G3-vs-evidence Ridge.

The *architectures* are compared on the same fresh-seed grouped holdouts.
This does NOT reuse the frozen historical G3 coefficients and is NOT a
prospective frozen-policy confirmation. Neither model trains on heldout labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..g5_fresh_contract import (
    DATASETS, SEEDS, CHECKPOINTS, HORIZONS, HORIZON_STEPS, ACTIONS,
    COST_WEIGHT, REFERENCE_SECONDS)
from ..g5_evidence_features import FEATURE_NAMES as PROPOSAL_NAMES
from .audit_pilot import audit_pilot
from .g5_evidence_ablation import JoinedDecision, fit_augmented, fold_result, summarize
from .value_predictor import (
    Decision, STATE_KEYS, FEATURE_NAMES as BASE_NAMES, RIDGE_ALPHA, TARGET_ACTIONS)


def load_studies(root: Path) -> tuple[list[JoinedDecision], str, dict]:
    decisions = []
    seen_checkpoints = set()
    commits = set()
    per_cohort = []
    for ds in DATASETS:
        for seed in SEEDS:
            directory = Path(root) / f"g5_{ds}_seed{seed}"
            manifest = json.loads(
                (directory / "g5a_cohort_manifest.json").read_text(encoding="utf-8"))
            design = manifest["design"]
            if (manifest.get("schema") != "awn_g5a_same_checkpoint_cohort_v1"
                    or not manifest["passed"] or manifest["dataset"] != ds
                    or manifest["seed"] != seed
                    or manifest["git_sha"] != design["git_sha"]
                    or design["dataset"] != ds or design["seed"] != seed
                    or design["checkpoint_epochs"] != list(CHECKPOINTS)
                    or design["horizon_steps"] != HORIZON_STEPS
                    or design["horizons"] != list(HORIZONS)
                    or design["actions"] != list(ACTIONS)
                    or design["cost_weight"] != COST_WEIGHT
                    or design["reference_seconds"] != REFERENCE_SECONDS
                    or design["test_examples_used"] != 0
                    or design["label_source"] !=
                    "paired_forks_of_exact_predecision_feature_snapshot"
                    or manifest["test_examples_used"] != 0
                    or manifest["split"]["test_examples_used"] != 0
                    or manifest["feature_names"] != list(PROPOSAL_NAMES)
                    or manifest["final_horizon_rollouts"] != 16
                    or manifest["all_horizon_observations"] != 48
                    or len(manifest["checkpoints"]) != 4):
                raise AssertionError("Missing or invalid new-seed cohort")
            commits.add(manifest["git_sha"])
            for idx, cp in enumerate(manifest["checkpoints"]):
                epoch = CHECKPOINTS[idx]
                cid = cp["checkpoint_id"]
                if cp["epoch"] != epoch or cid in seen_checkpoints or not cp["audited"]:
                    raise AssertionError("Duplicated or reordered checkpoint")
                seen_checkpoints.add(cid)
                directory_name = f"epoch{epoch:04d}"
                if cp["pilot_subdir"] != directory_name:
                    raise AssertionError("Checkpoint audit path mismatch")
                cp_dir = directory / directory_name
                pre = json.loads((cp_dir / "preaction.json").read_text(encoding="utf-8"))
                if (set(pre) != {
                    "checkpoint_id", "dataset", "seed", "epoch", "state",
                    "feature_names", "proposal_features",
                    "feature_compute_seconds", "future_outcomes_available"
                } or pre["checkpoint_id"] != cid or pre["dataset"] != ds
                        or pre["seed"] != seed or pre["epoch"] != epoch
                        or pre["state"] != cp["decision_state"]
                        or pre["proposal_features"] != cp["proposal_features"]
                        or pre["feature_names"] != list(PROPOSAL_NAMES)
                        or pre["future_outcomes_available"] is not False
                        or pre["feature_compute_seconds"] != cp["feature_compute_seconds"]):
                    raise AssertionError("Preaction evidence changed after labels")
                state = pre["state"]
                if set(state) != set(STATE_KEYS):
                    raise AssertionError("Invalid G3 baseline state features")
                hist = manifest["baseline_training_history"][epoch - 1]
                if (state["epoch"] != epoch
                        or state["physical_width"] != hist["physical_width"]
                        or state["learned_width"] != hist["learned_width"]
                        or state["validation_ce"] != hist["validation_ce"]
                        or state["train_elbo_loss"] != hist["mean_train_elbo_loss"]
                        or state["training_steps_so_far"] != sum(
                            h["training_steps"] for h in
                            manifest["baseline_training_history"][:epoch])):
                    raise AssertionError("Features don't belong to native AWN checkpoint")
                feat = pre["proposal_features"]
                if set(feat) != set(PROPOSAL_NAMES):
                    raise AssertionError("Unapproved future-dependent feature")
                extra = tuple(float(feat[k]) for k in PROPOSAL_NAMES)
                base = tuple(float(state[k]) for k in STATE_KEYS) + (
                    float(ds == "SpiralHard"),)
                if (not np.isfinite(extra).all() or not np.isfinite(base).all()
                        or not 0 <= float(pre["feature_compute_seconds"])
                        or any(not (0 <= v <= 1) for v in extra)):
                    raise AssertionError("Invalid numeric predecision features")
                audit = audit_pilot(cp_dir)
                saved_audit = json.loads((cp_dir / "audit.json").read_text(encoding="utf-8"))
                if audit != saved_audit:
                    raise AssertionError("G5A raw treatment data fail re-audit")
                if (not audit["passed"] or audit["git_sha"] != manifest["git_sha"]
                        or audit["checkpoint_id"] != cid or audit["dataset"] != ds
                        or audit["seed"] != seed or audit["test_examples_used"] != 0
                        or audit["paired_horizons"] != list(HORIZONS)
                        or audit["observations"] != 12 or audit["labels"] != 12):
                    raise AssertionError("Missing paired treatment outcomes")
                rows = audit["final_horizon_comparison"]
                if len(rows) != len(ACTIONS) or {r["action"] for r in rows} != set(ACTIONS):
                    raise AssertionError("Missing a candidate action")
                outcomes = {}
                for r in rows:
                    if (r["gate_mode"] != "positional"
                            or r["optimizer_steps"] != HORIZON_STEPS):
                        raise AssertionError("G5A intervention semantics changed")
                    outcomes[r["action"]] = {
                        "net_value": float(r["net_value"]),
                        "delta_risk": float(r["delta_risk"]),
                        "no_op": bool(r["no_op"]),
                        "delta_online_seconds": float(r["delta_online_seconds"]),
                    }
                if outcomes["skip"]["net_value"] != 0 or not outcomes["skip"]["no_op"]:
                    raise AssertionError("Untreated reference is missing")
                decisions.append(JoinedDecision(
                    Decision(ds, seed, epoch, cid, base, outcomes), extra))
            per_cohort.append({"dataset": ds, "seed": seed, "checkpoints": 4})
    if len(commits) != 1 or len(decisions) != 24 or len(seen_checkpoints) != 24:
        raise AssertionError("Incomplete or mixed-code G5A study")
    return decisions, next(iter(commits)), {"cohorts": per_cohort}


def evaluate(studies: list[JoinedDecision], sha: str) -> dict:
    if len(studies) != 24 or len({d.base.checkpoint_id for d in studies}) != 24:
        raise AssertionError("Not 24 independent checkpoint contexts")
    seed_folds = [
        fold_result(
            f"heldout_seed_{seed}",
            [d for d in studies if d.base.seed != seed],
            [d for d in studies if d.base.seed == seed],
        ) for seed in SEEDS
    ]
    dataset_folds = [
        fold_result(
            f"heldout_dataset_{ds}",
            [d for d in studies if d.base.dataset != ds],
            [d for d in studies if d.base.dataset == ds],
        ) for ds in DATASETS
    ]
    return {
        "schema": "awn_g5a_same_checkpoint_fresh_seed_ablation_v1",
        "source_code_sha": sha, "cohorts": len(DATASETS) * len(SEEDS),
        "decisions": len(studies), "paired_final_horizon_rollouts": len(studies) * 4,
        "horizons": list(HORIZONS), "test_examples_used": 0,
        "baseline": "G3 architecture retrained per training fold (NOT frozen G3 coefficients)",
        "augmented": "G3 architecture plus exactly 8 preaction evidence/proposal features",
        "fixed_ridge_alpha": RIDGE_ALPHA, "baseline_features": list(BASE_NAMES),
        "proposal_features": list(PROPOSAL_NAMES),
        "seed_holdout": {
            "folds": seed_folds, "summary": summarize(seed_folds),
        },
        "dataset_holdout": {
            "folds": dataset_folds, "summary": summarize(dataset_folds),
        },
        "full_cohort_augmented_candidate_not_independently_validated": fit_augmented(studies),
        "limitations": (
            "Exploratory grouped holdouts on two same-family synthetic datasets "
            "and three seeds. Added G/W calibration feature overhead is NOT "
            "charged to decision policies here. Do not interpret any positive "
            "net gain as an online end-to-end gain or as G4 prospective confirmation. "
            "Prior historical G2 replay failed exact-state matching on SpiralHard "
            "and is deliberately NOT joined to G2 labels."
        ), "passed": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    studies, sha, provenance = load_studies(args.root)
    report = evaluate(studies, sha)
    report["provenance"] = provenance
    if args.output.exists():
        raise FileExistsError("G5A evaluation output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "cohorts": report["cohorts"],
        "checkpoint_decisions": report["decisions"],
        "seed_holdout": report["seed_holdout"]["summary"],
        "dataset_holdout": report["dataset_holdout"]["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
