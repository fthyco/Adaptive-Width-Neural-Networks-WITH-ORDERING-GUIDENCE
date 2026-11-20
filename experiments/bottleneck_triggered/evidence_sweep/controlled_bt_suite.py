"""Reproducible sequential grid runner and paired AWN BT comparison summary."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

from .controlled_bt import GATE_MODES, METHODS, run_experiment
from .train_screen import DEFAULT_EPOCHS


def _names(csv_string: str, allowed: tuple[str, ...] | list[str]) -> list[str]:
    names = [name.strip() for name in csv_string.split(",") if name.strip()]
    if not names or len(set(names)) != len(names) or any(n not in allowed for n in names):
        raise ValueError("Unknown/duplicate/empty selection: " + csv_string)
    return names


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def paired_rows(rows: list[dict]) -> list[dict]:
    """Pair by dataset, seed and schedule, never by a different test split."""
    baselines = {(r["dataset"], r["seed"], r["schedule"]): r
                 for r in rows if r["method"] == "identity" and r["gate_mode"] == "positional"}
    result = []
    for r in rows:
        output = dict(r)
        baseline = baselines.get((r["dataset"], r["seed"], r["schedule"]))
        output["paired_baseline_available"] = baseline is not None
        output["test_accuracy_delta_vs_identity"] = (
            r["final_test_accuracy"] - baseline["final_test_accuracy"]
            if baseline else None)
        output["test_ce_delta_vs_identity"] = (
            r["final_test_ce"] - baseline["final_test_ce"]
            if baseline else None)
        result.append(output)
    return result


def _save(rows: list[dict], out: Path, failures: list[dict], metadata: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    paired = paired_rows(rows)
    payload = {"schema": "controlled_bt_grid_v1", "metadata": metadata,
               "completed_runs": len(rows), "failures": failures, "runs": paired}
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if paired:
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            w = csv.DictWriter(handle, fieldnames=list(paired[0].keys()))
            w.writeheader()
            w.writerows(paired)


def run_suite(output: Path, datasets: list[str], seeds: list[int],
              methods: list[str], gate_modes: list[str], schedules: list[str],
              *, epochs_override: int | None = None, fixed_epochs: tuple[int, ...] = (10, 20),
              max_interventions: int = 2, device: str = "cpu",
              learning_rate: float = 0.01,
              max_calibration_batches: int | None = None,
              resume: bool = False, keep_going: bool = False,
              save_model: bool = False) -> dict:
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be nonempty unique integers")
    sha = git_sha()
    metadata = {
        "git_sha": sha, "datasets": datasets, "seeds": seeds, "methods": methods,
        "gate_modes": gate_modes, "schedules": schedules,
        "epochs_override": epochs_override,
        "fixed_epochs": fixed_epochs, "max_interventions": max_interventions,
        "device": device, "learning_rate": learning_rate,
        "max_calibration_batches": max_calibration_batches,
        "save_model": save_model,
        "note": "Train from scratch for every configuration; only fixed schedules are event-matched",
    }
    rows, failures = [], []
    for dataset in datasets:
        for seed in seeds:
            for schedule in schedules:
                # Exactly one unchanged AWN baseline per paired dataset/seed/schedule.
                grid = [("identity", "positional")]
                grid += [(m, gate) for m in methods if m != "identity" for gate in gate_modes]
                for method, mode in grid:
                    epochs = epochs_override or DEFAULT_EPOCHS[dataset]
                    path = output / dataset / ("seed%d" % seed) / schedule / method / mode
                    manifest_path = path / "run_manifest.json"
                    try:
                        if resume and manifest_path.exists():
                            rec = json.loads(manifest_path.read_text(encoding="utf-8"))
                            keys = {"dataset": dataset, "seed": seed, "schedule": schedule,
                                    "method": method, "gate_mode": mode, "epochs": epochs,
                                    "learning_rate": learning_rate, "device": device,
                                    "max_interventions": max_interventions,
                                    "max_calibration_batches": max_calibration_batches,
                                    "save_model": save_model, "git_sha": sha,
                                    "fixed_epochs": sorted(fixed_epochs)}
                            if any(rec.get(k) != v for k, v in keys.items()):
                                raise ValueError("Existing manifest config mismatch; refuse stale resume")
                        else:
                            rec = run_experiment(
                                dataset=dataset, seed=seed, method=method, gate_mode=mode,
                                schedule=schedule, epochs=epochs, output=path,
                                fixed_epochs=fixed_epochs, max_interventions=max_interventions,
                                device=device, learning_rate=learning_rate,
                                max_calibration_batches=max_calibration_batches,
                                save_model=save_model)
                        row = {
                            "dataset": dataset, "seed": seed, "schedule": schedule,
                            "method": method, "gate_mode": mode, "epochs": epochs,
                            "n_interventions": len(rec["interventions"]),
                            "intervention_epochs": "|".join(
                                str(e["epoch"]) for e in rec["interventions"]),
                            "final_width": rec["final_width"],
                            "final_test_accuracy": rec["final_test_accuracy"],
                            "final_test_ce": rec["final_test_ce"],
                            "final_validation_accuracy": rec["epochs_log"][-1]["validation_accuracy"],
                            "final_validation_ce": rec["epochs_log"][-1]["validation_ce"],
                            "optimizer_steps": rec["optimizer_steps"],
                            "training_examples": rec["training_examples"],
                            "evidence_seconds": sum(i["evidence_seconds"] for i in rec["interventions"]),
                            "permutation_seconds": sum(i["permutation_seconds"] for i in rec["interventions"]),
                            "elapsed_seconds": rec["elapsed_seconds"],
                            "manifest_path": str(manifest_path),
                        }
                        rows.append(row)
                        print(json.dumps({"complete": row["manifest_path"],
                                          "test_acc": row["final_test_accuracy"],
                                          "interventions": row["n_interventions"]}),
                              flush=True)
                    except Exception as exc:
                        failure = {"dataset": dataset, "seed": seed, "schedule": schedule,
                                   "method": method, "gate_mode": mode,
                                   "error": "%s: %s" % (type(exc).__name__, exc)}
                        failures.append(failure)
                        _save(rows, output, failures, metadata)
                        if not keep_going:
                            raise
                    _save(rows, output, failures, metadata)
    return {"completed": len(rows), "failures": len(failures), "git_sha": sha}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="DoubleMoon")
    p.add_argument("--seeds", default="42,456")
    p.add_argument("--methods", default="identity,random,W,G,J,I,GI,S,S_tie_shuffle")
    p.add_argument("--gate-modes", default="positional,neuron_coupled")
    p.add_argument("--schedules", default="fixed")
    p.add_argument("--epochs", type=int)
    p.add_argument("--fixed-epochs", default="10,20")
    p.add_argument("--max-interventions", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--max-calibration-batches", type=int)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    datasets = _names(args.datasets, sorted(DEFAULT_EPOCHS))
    seeds = [int(v.strip()) for v in args.seeds.split(",") if v.strip()]
    methods = _names(args.methods, METHODS)
    gate_modes = _names(args.gate_modes, GATE_MODES)
    schedules = _names(args.schedules, ("fixed", "bt"))
    epochs = tuple(int(x) for x in args.fixed_epochs.split(",") if x.strip())
    result = run_suite(
        args.output, datasets, seeds, methods, gate_modes, schedules,
        epochs_override=args.epochs, fixed_epochs=epochs,
        max_interventions=args.max_interventions, device=args.device,
        learning_rate=args.learning_rate,
        max_calibration_batches=args.max_calibration_batches,
        resume=args.resume, keep_going=args.keep_going,
        save_model=args.save_model)
    print(json.dumps(result, indent=2))
    return int(result["failures"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
