"""Predeclared Spiral fixed-event AWN replication: 5 seeds x 9 configurations.

This is a follow-up on previously inspected Spiral results. It is not a fresh,
independent confirmatory test. No core AWN code is changed by this module.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import math
from pathlib import Path
import random
from statistics import mean, stdev

DATASET = "Spiral"
SEEDS = (42, 123, 456, 789, 1000)
METHODS = ("identity", "random", "W", "G", "GI")
MODES = ("positional", "neuron_coupled")
EVENT_EPOCHS = (15, 30)
EPOCHS = 50
EXPECTED_RUNS = len(SEEDS) * (1 + (len(METHODS) - 1) * len(MODES))

# Left minus right; registered before running the fixed-event study.
CONTRASTS = (
    ("primary_G_vs_random_positional", ("G", "positional"), ("random", "positional")),
    ("secondary_G_vs_W_positional", ("G", "positional"), ("W", "positional")),
    ("secondary_G_vs_GI_positional", ("G", "positional"), ("GI", "positional")),
    ("secondary_G_vs_random_coupled", ("G", "neuron_coupled"), ("random", "neuron_coupled")),
    ("secondary_G_positional_vs_coupled", ("G", "positional"), ("G", "neuron_coupled")),
    ("secondary_random_positional_vs_coupled", ("random", "positional"), ("random", "neuron_coupled")),
    ("secondary_W_positional_vs_coupled", ("W", "positional"), ("W", "neuron_coupled")),
    ("secondary_GI_positional_vs_coupled", ("GI", "positional"), ("GI", "neuron_coupled")),
    ("context_G_positional_vs_AWN", ("G", "positional"), ("identity", "positional")),
    ("context_random_positional_vs_AWN", ("random", "positional"), ("identity", "positional")),
)


def grid() -> tuple[tuple[int, str, str], ...]:
    return tuple((seed, method, gate) for seed in SEEDS
                 for method, gate in [("identity", "positional"),
                                      *product(METHODS[1:], MODES)])


def plan(source_sha: str, *, device: str, learning_rate: float,
         save_model: bool, max_calibration_batches: int | None) -> dict:
    return {
        "schema": "spiral_fixed_causal_v1",
        "source_sha": source_sha,
        "dataset": DATASET,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "gate_modes": list(MODES),
        "schedule": "fixed",
        "fixed_epochs": list(EVENT_EPOCHS),
        "epochs": EPOCHS,
        "max_interventions": len(EVENT_EPOCHS),
        "expected_runs": EXPECTED_RUNS,
        "device": device,
        "learning_rate": learning_rate,
        "save_model": save_model,
        "max_calibration_batches": max_calibration_batches,
        "primary_endpoint": "final_test_accuracy",
        "secondary_endpoint": "final_test_ce",
        "primary_contrast": CONTRASTS[0][0],
        "interpretation": "Previously examined seeds/test partition: controlled replication, not fresh confirmation",
    }


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _verify_or_create_plan(output: Path, expected: dict, resume: bool) -> None:
    target = output / "study_plan.json"
    if target.exists():
        if _json(target) != expected:
            raise ValueError("Study plan/source configuration mismatch: choose a fresh output directory")
        if not resume:
            raise ValueError("Study output exists: use --resume to continue without overwriting")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Non-empty output without study plan; refuse to mix experiments")
        _atomic_json(target, expected)


def run(output: Path, *, device: str, learning_rate: float,
        save_model: bool, max_calibration_batches: int | None,
        resume: bool) -> dict:
    from .controlled_bt_suite import git_sha, run_suite
    study = plan(git_sha(), device=device, learning_rate=learning_rate,
                 save_model=save_model, max_calibration_batches=max_calibration_batches)
    if study["source_sha"] == "unknown":
        raise RuntimeError("Git SHA cannot be verified")
    _verify_or_create_plan(output, study, resume)
    result = run_suite(output, [DATASET], list(SEEDS), list(METHODS), list(MODES),
                       ["fixed"], epochs_override=EPOCHS, fixed_epochs=EVENT_EPOCHS,
                       max_interventions=2, device=device, learning_rate=learning_rate,
                       max_calibration_batches=max_calibration_batches,
                       save_model=save_model, resume=resume, keep_going=False)
    # Strict audit is run only if the complete grid returned successfully.
    audit(output)
    return result


def audit(output: Path) -> dict:
    study = _json(output / "study_plan.json")
    if study.get("schema") != "spiral_fixed_causal_v1" or study.get("expected_runs") != EXPECTED_RUNS:
        raise ValueError("Missing or invalid study plan")
    payload = _json(output / "summary.json")
    rows = payload.get("runs", [])
    problems: list[str] = []
    if payload.get("failures"):
        problems.append("Suite reports failed runs")
    if len(rows) != EXPECTED_RUNS:
        problems.append(f"Expected {EXPECTED_RUNS} runs, found {len(rows)}")
    by_key = {}
    for r in rows:
        key = (int(r["seed"]), r["method"], r["gate_mode"])
        if key in by_key:
            problems.append(f"Duplicate configuration: {key}")
        by_key[key] = r
    if set(by_key) != set(grid()):
        problems.append(f"Grid mismatch; missing={sorted(set(grid()) - set(by_key))}, "
                        f"unexpected={sorted(set(by_key) - set(grid()))}")
    manifest_data = {}
    for key, r in sorted(by_key.items()):
        seed, method, mode = key
        p = output / DATASET / f"seed{seed}" / "fixed" / method / mode / "run_manifest.json"
        if not p.exists():
            problems.append(f"Missing manifest {p}")
            continue
        m = _json(p)
        manifest_data[key] = m
        for name, expected in (("dataset", DATASET), ("seed", seed),
                               ("method", method), ("gate_mode", mode),
                               ("schedule", "fixed"), ("epochs", EPOCHS),
                               ("fixed_epochs", list(EVENT_EPOCHS)),
                               ("max_interventions", 2),
                               ("test_examples_used_for_selection", 0),
                               ("test_evaluated_after_training", True),
                               ("git_sha", study["source_sha"])):
            if m.get(name) != expected:
                problems.append(f"{key} mismatch {name}: {m.get(name)!r} != {expected!r}")
        actual = [int(x["epoch"]) for x in m.get("interventions", [])]
        expected_events = [] if method == "identity" else list(EVENT_EPOCHS)
        if actual != expected_events:
            problems.append(f"{key} event epochs {actual} != {expected_events}")
        if len(m.get("epochs_log", [])) != EPOCHS:
            problems.append(f"{key} incomplete epoch logs")
        if int(r["optimizer_steps"]) != int(m.get("optimizer_steps", -1)):
            problems.append(f"{key} summary/manifest steps mismatch")
        if int(r["training_examples"]) != int(m.get("training_examples", -1)):
            problems.append(f"{key} summary/manifest examples mismatch")
        for metric in ("final_test_accuracy", "final_test_ce"):
            summary_value = float(r[metric])
            manifest_value = float(m.get(metric, float("nan")))
            if not (math.isfinite(summary_value) and math.isfinite(manifest_value)):
                problems.append(f"{key} non-finite {metric}")
            elif abs(summary_value - manifest_value) > 1e-9:
                problems.append(f"{key} summary/manifest {metric} mismatch")
        if mode == "neuron_coupled":
            for ev in m["interventions"]:
                d = float(ev["full_width_output_max_diff"])
                if not math.isfinite(d) or d > 1e-4:
                    problems.append(f"{key} failed coupled full-width equivalence")
        for ev in m["interventions"]:
            d = float(ev["no_gate_output_max_diff"])
            if not math.isfinite(d) or d > 1e-4:
                problems.append(f"{key} failed no-gate equivalence")
    for seed in SEEDS:
        base = manifest_data.get((seed, "identity", "positional"))
        if base is None:
            problems.append(f"Missing baseline: {seed}")
            continue
        for method in METHODS[1:]:
            for mode in MODES:
                item = manifest_data.get((seed, method, mode))
                if item is None:
                    continue
                for k in ("optimizer_steps", "training_examples", "split"):
                    if item.get(k) != base.get(k):
                        problems.append(f"Seed {seed} {method}/{mode}: {k} differs from baseline")
    result = {"schema": "spiral_fixed_causal_audit_v1", "expected_runs": EXPECTED_RUNS,
              "observed_runs": len(rows), "checked_manifests": len(manifest_data),
              "pass": not problems, "problems": problems}
    _atomic_json(output / "audit.json", result)
    if problems:
        raise ValueError("Fixed-event audit failed: " + "; ".join(problems[:10]))
    return result


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("No records to export")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _effect(values: list[float]) -> dict:
    n = len(values)
    if n != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} paired seeds, received {n}")
    observed = mean(values)
    # Exact two-sided paired sign-flip randomization p under exchangeability.
    # With five pairs, the minimum attainable two-sided p is 2/32 = 0.0625.
    flips = (mean([x * s for x, s in zip(values, signs)])
             for signs in product((-1, 1), repeat=n))
    p = sum(abs(t) >= abs(observed) - 1e-12 for t in flips) / (2 ** n)
    rng = random.Random(1729)
    boots = sorted(mean([values[rng.randrange(n)] for _ in range(n)])
                   for _ in range(10000))
    return {"n_seeds": n, "mean": observed,
            "std": stdev(values), "positive": sum(x > 1e-10 for x in values),
            "negative": sum(x < -1e-10 for x in values),
            "bootstrap_95_low": boots[249], "bootstrap_95_high": boots[9749],
            "exact_two_sided_sign_flip_p": p}


def analyze(output: Path) -> dict:
    audit(output)
    rows = _json(output / "summary.json")["runs"]
    by_key = {(int(x["seed"]), x["method"], x["gate_mode"]): x for x in rows}
    pair_rows, summary_rows = [], []
    metrics = ("final_test_accuracy", "final_test_ce",
               "final_validation_accuracy", "final_validation_ce")
    for name, left, right in CONTRASTS:
        differences = {metric: [] for metric in metrics}
        for seed in SEEDS:
            a, b = by_key[(seed, *left)], by_key[(seed, *right)]
            row = {"contrast": name, "seed": seed,
                   "left_method": left[0], "left_gate_mode": left[1],
                   "right_method": right[0], "right_gate_mode": right[1]}
            for metric in metrics:
                delta = float(a[metric]) - float(b[metric])
                row[metric + "_diff"] = delta
                differences[metric].append(delta)
            pair_rows.append(row)
        sums = {metric: _effect(vals) for metric, vals in differences.items()}
        summary_rows.append({"contrast": name, "n_seeds": len(SEEDS),
                             "accuracy_diff_mean": sums["final_test_accuracy"]["mean"],
                             "accuracy_diff_std": sums["final_test_accuracy"]["std"],
                             "accuracy_diff_ci_low": sums["final_test_accuracy"]["bootstrap_95_low"],
                             "accuracy_diff_ci_high": sums["final_test_accuracy"]["bootstrap_95_high"],
                             "accuracy_diff_exact_signflip_p": sums["final_test_accuracy"]["exact_two_sided_sign_flip_p"],
                             "accuracy_positive_seeds": sums["final_test_accuracy"]["positive"],
                             "ce_diff_mean": sums["final_test_ce"]["mean"],
                             "ce_diff_exact_signflip_p": sums["final_test_ce"]["exact_two_sided_sign_flip_p"],
                             "validation_accuracy_diff_mean": sums["final_validation_accuracy"]["mean"],
                             "validation_ce_diff_mean": sums["final_validation_ce"]["mean"]})
    trajectories, events = [], []
    for seed, method, mode in grid():
        manifest_path = (output / DATASET / f"seed{seed}" / "fixed" / method / mode /
                         "run_manifest.json")
        m = _json(manifest_path)
        for ep in m["epochs_log"]:
            trajectories.append({"seed": seed, "method": method, "gate_mode": mode,
                                 "epoch": ep["epoch"], "validation_accuracy": ep["validation_accuracy"],
                                 "validation_ce": ep["validation_ce"], "width": ep["width"],
                                 "intervention_count_so_far": ep.get("intervention", "")})
        for ev in m["interventions"]:
            events.append({"seed": seed, "method": method, "gate_mode": mode,
                           "epoch": ev["epoch"],
                           "full_width_output_max_diff": ev["full_width_output_max_diff"],
                           "no_gate_output_max_diff": ev["no_gate_output_max_diff"],
                           "evidence_seconds": ev["evidence_seconds"],
                           "permutation_seconds": ev["permutation_seconds"]})
    analysis_dir = output / "causal_analysis"
    _write_csv(analysis_dir / "paired_seed_deltas.csv", pair_rows)
    _write_csv(analysis_dir / "paired_contrast_summary.csv", summary_rows)
    _write_csv(analysis_dir / "validation_trajectories.csv", trajectories)
    _write_csv(analysis_dir / "intervention_events.csv", events)
    result = {"schema": "spiral_fixed_causal_analysis_v1", "n_runs": EXPECTED_RUNS,
              "contrasts": list(row["contrast"] for row in summary_rows),
              "primary_contrast": CONTRASTS[0][0],
              "primary_result": summary_rows[0],
              "study_limitations": [
                  "Five previously analyzed seeds and previously inspected test split; not fresh confirmation",
                  "Sign-flip inference requires exchangeability; multiplicity not adjusted",
                  "Epoch validation is measured before that epoch's intervention; no immediate post-event validation CE claimed",
                  "Neuron-coupled gate mode is a counterfactual, not original AWN variational semantics"],
              "records": summary_rows}
    _atomic_json(analysis_dir / "analysis.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Execute locked 45-run Spiral grid")
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--learning-rate", type=float, default=0.01)
    p_run.add_argument("--max-calibration-batches", type=int)
    p_run.add_argument("--save-model", action="store_true")
    p_run.add_argument("--resume", action="store_true")
    for name in ("audit", "analyze"):
        sub.add_parser(name).add_argument("--output", type=Path, required=True)
    sub.add_parser("plan", help="Print immutable run plan without executing")
    args = parser.parse_args(argv)
    if args.command == "plan":
        print(json.dumps({"expected_runs": EXPECTED_RUNS,
                          "grid": grid(), "contrasts": CONTRASTS}, indent=2))
        return 0
    if args.command == "run":
        result = run(args.output, device=args.device,
                     learning_rate=args.learning_rate,
                     max_calibration_batches=args.max_calibration_batches,
                     save_model=args.save_model, resume=args.resume)
    elif args.command == "audit":
        result = audit(args.output)
    else:
        result = analyze(args.output)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
