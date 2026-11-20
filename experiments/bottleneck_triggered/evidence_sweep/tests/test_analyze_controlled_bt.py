"""Verify paired analysis never pairs different seeds or schedules."""
import csv

from experiments.bottleneck_triggered.evidence_sweep.analyze_controlled_bt import analyze


def _write(path, records):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _item(seed, method, mode, accuracy, ce=0.5):
    return {
        "dataset": "DoubleMoon", "seed": seed, "schedule": "fixed",
        "method": method, "gate_mode": mode, "final_test_accuracy": accuracy,
        "final_test_ce": ce, "optimizer_steps": 30,
        "training_examples": 240, "evidence_seconds": 0.04,
        "permutation_seconds": 0.01, "n_interventions": 2,
    }


def test_descriptive_paired_analysis(tmp_path):
    src = tmp_path / "summary.csv"
    _write(src, [
        _item(42, "identity", "positional", 0.70),
        _item(42, "GI", "positional", 0.73),
        _item(42, "GI", "neuron_coupled", 0.78),
        _item(123, "identity", "positional", 0.82),
        _item(123, "GI", "positional", 0.80),
        _item(123, "GI", "neuron_coupled", 0.83),
    ])
    result = analyze(src, tmp_path / "analysis")
    assert result["mismatches"] == []
    assert len(result["gate_mode_summary"]) == 1
    row = result["gate_mode_summary"][0]
    assert row["n_paired_seeds"] == 2
    assert abs(row["coupled_minus_positional_test_accuracy_mean"] - 0.04) < 1e-10
    candidate = next(r for r in result["candidate_summary"]
                     if r["method"] == "GI" and r["gate_mode"] == "neuron_coupled")
    assert abs(candidate["test_accuracy_delta_mean"] - 0.045) < 1e-10
    assert (tmp_path / "analysis" / "candidate_paired_summary.csv").exists()


def test_detect_budget_mismatch(tmp_path):
    src = tmp_path / "summary.csv"
    a = _item(42, "identity", "positional", 0.70)
    b = _item(42, "GI", "neuron_coupled", 0.75)
    b["optimizer_steps"] = 29
    _write(src, [a, b])
    result = analyze(src, tmp_path / "analysis")
    assert len(result["mismatches"]) == 1
    assert result["mismatches"][0]["reason"] == "training_budget_mismatch"
