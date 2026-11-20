"""Aggregate paired AWN live BT outcomes without selecting by final test data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _stats(values: list[float], seed: int = 12345) -> dict:
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("No finite paired observations to summarize")
    rng = np.random.default_rng(seed)
    # Descriptive uncertainty only: 5 seeds cannot establish a general law.
    bootstrap = a[rng.integers(len(a), size=(10000, len(a)))].mean(axis=1)
    lo, hi = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": len(a),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "bootstrap_95_low": float(lo),
        "bootstrap_95_high": float(hi),
        "positive": int(np.sum(a > 1e-10)),
        "zero": int(np.sum(np.abs(a) <= 1e-10)),
        "negative": int(np.sum(a < -1e-10)),
    }


def analyze(summary_csv: Path, output: Path) -> dict:
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No controlled BT runs in CSV")
    numeric = ("seed", "final_test_accuracy", "final_test_ce", "optimizer_steps",
               "training_examples", "evidence_seconds", "permutation_seconds",
               "n_interventions")
    for row in rows:
        for k in numeric:
            row[k] = float(row[k]) if k != "seed" else int(row[k])
    keys = ["dataset", "seed", "schedule", "method", "gate_mode"]
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in seen:
            raise ValueError("Duplicate run in summary: " + str(key))
        seen.add(key)
    baselines = {
        (r["dataset"], r["seed"], r["schedule"]): r for r in rows
        if r["method"] == "identity" and r["gate_mode"] == "positional"
    }
    by_candidate = defaultdict(list)
    by_mode = defaultdict(dict)
    mismatches = []
    for r in rows:
        base_key = (r["dataset"], r["seed"], r["schedule"])
        baseline = baselines.get(base_key)
        if baseline is None:
            mismatches.append({"run": base_key, "reason": "missing_identity_baseline"})
            continue
        if (r["optimizer_steps"], r["training_examples"]) != (
                baseline["optimizer_steps"], baseline["training_examples"]):
            mismatches.append({"run": base_key, "method": r["method"],
                               "gate_mode": r["gate_mode"], "reason": "training_budget_mismatch"})
            continue
        by_candidate[(r["dataset"], r["schedule"], r["method"], r["gate_mode"])].append(
            {"seed": r["seed"],
             "accuracy_delta": r["final_test_accuracy"] - baseline["final_test_accuracy"],
             "ce_delta": r["final_test_ce"] - baseline["final_test_ce"],
             "evidence_seconds": r["evidence_seconds"],
             "permutation_seconds": r["permutation_seconds"],
             "n_interventions": r["n_interventions"]})
        by_mode[(r["dataset"], r["seed"], r["schedule"], r["method"])][r["gate_mode"]] = r

    group_records = []
    for key, records in sorted(by_candidate.items()):
        dataset, schedule, method, mode = key
        group_records.append({
            "dataset": dataset, "schedule": schedule, "method": method, "gate_mode": mode,
            "seeds": "|".join(str(e["seed"]) for e in sorted(records, key=lambda x: x["seed"])),
            "n_seeds": len(records),
            "test_accuracy_delta_mean": _stats([r["accuracy_delta"] for r in records])["mean"],
            "test_accuracy_delta_std": _stats([r["accuracy_delta"] for r in records])["std"],
            "test_accuracy_delta_ci_low": _stats([r["accuracy_delta"] for r in records])["bootstrap_95_low"],
            "test_accuracy_delta_ci_high": _stats([r["accuracy_delta"] for r in records])["bootstrap_95_high"],
            "positive_seeds": _stats([r["accuracy_delta"] for r in records])["positive"],
            "test_ce_delta_mean": _stats([r["ce_delta"] for r in records])["mean"],
            "evidence_seconds_mean": _stats([r["evidence_seconds"] for r in records])["mean"],
            "permutation_seconds_mean": _stats([r["permutation_seconds"] for r in records])["mean"],
            "interventions_mean": _stats([r["n_interventions"] for r in records])["mean"],
        })

    mode_differences = defaultdict(list)
    for (dataset, seed, schedule, method), pair in by_mode.items():
        if "positional" in pair and "neuron_coupled" in pair:
            diff = pair["neuron_coupled"]["final_test_accuracy"] - pair["positional"]["final_test_accuracy"]
            mode_differences[(dataset, schedule, method)].append({"seed": seed, "diff": diff})
    mode_records = []
    for (dataset, schedule, method), values in sorted(mode_differences.items()):
        summary = _stats([v["diff"] for v in values])
        mode_records.append({
            "dataset": dataset, "schedule": schedule, "method": method,
            "n_paired_seeds": summary["n"],
            "coupled_minus_positional_test_accuracy_mean": summary["mean"],
            "coupled_minus_positional_test_accuracy_std": summary["std"],
            "ci_low": summary["bootstrap_95_low"], "ci_high": summary["bootstrap_95_high"],
            "positive_seeds": summary["positive"], "negative_seeds": summary["negative"],
        })
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "candidate_paired_summary.csv", group_records)
    _write_csv(output / "gating_mode_paired_summary.csv", mode_records)
    result = {
        "schema": "controlled_bt_paired_analysis_v1",
        "n_runs": len(rows),
        "n_candidate_groups": len(group_records),
        "n_mode_groups": len(mode_records),
        "mismatches": mismatches,
        "interpretation": "descriptive paired test outcomes; avoid selection from test and overclaiming significance",
        "candidate_summary": group_records,
        "gate_mode_summary": mode_records,
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = analyze(args.summary, args.output)
    print(json.dumps({k: result[k] for k in ("n_runs", "n_candidate_groups",
                                              "n_mode_groups", "mismatches")}, indent=2))
    return int(bool(result["mismatches"]))


if __name__ == "__main__":
    raise SystemExit(main())
