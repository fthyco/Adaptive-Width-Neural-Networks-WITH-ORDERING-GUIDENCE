"""Paired reporting and selection controls."""
import pytest

from experiments.bottleneck_triggered.evidence_sweep.controlled_bt_suite import (
    _names, paired_rows,
)


def test_paired_accuracy_and_ce_by_exact_seed_dataset_schedule():
    items = [
        {"dataset": "DoubleMoon", "seed": 42, "schedule": "fixed",
         "method": "identity", "gate_mode": "positional",
         "final_test_accuracy": 0.81, "final_test_ce": 0.47},
        {"dataset": "DoubleMoon", "seed": 42, "schedule": "fixed",
         "method": "GI", "gate_mode": "neuron_coupled",
         "final_test_accuracy": 0.86, "final_test_ce": 0.42},
        {"dataset": "DoubleMoon", "seed": 123, "schedule": "fixed",
         "method": "GI", "gate_mode": "neuron_coupled",
         "final_test_accuracy": 0.70, "final_test_ce": 0.60},
        {"dataset": "DoubleMoon", "seed": 42, "schedule": "bt",
         "method": "GI", "gate_mode": "neuron_coupled",
         "final_test_accuracy": 0.89, "final_test_ce": 0.39},
    ]
    rows = paired_rows(items)
    assert rows[0]["test_accuracy_delta_vs_identity"] == pytest.approx(0)
    assert rows[1]["test_accuracy_delta_vs_identity"] == pytest.approx(0.05)
    assert rows[1]["test_ce_delta_vs_identity"] == pytest.approx(-0.05)
    assert rows[2]["paired_baseline_available"] is False
    assert rows[3]["paired_baseline_available"] is False


def test_grid_selection_must_be_nonempty_unique_and_valid():
    assert _names("G,GI,S", ["G", "GI", "S"]) == ["G", "GI", "S"]
    for invalid in ("", "G,G", "G,NOPE"):
        with pytest.raises(ValueError):
            _names(invalid, ["G", "GI", "S"])
