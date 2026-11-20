"""Synthetic contract tests; they do not claim real AWN improvement."""
import csv
import json

import pytest

from experiments.bottleneck_triggered.evidence_sweep import spiral_fixed_causal as study


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fake_complete_grid(root):
    source = "a" * 40
    _write(root / "study_plan.json", study.plan(source, device="cpu",
            learning_rate=0.01, save_model=False, max_calibration_batches=None))
    runs = []
    for seed, method, mode in study.grid():
        baseline = 0.60 + seed / 100000
        boost = {"identity": 0, "random": 0.005, "W": 0.020,
                 "G": 0.040, "GI": 0.025}[method]
        if mode == "neuron_coupled":
            boost -= 0.015
        acc = baseline + boost
        events = [] if method == "identity" else [
            {"epoch": t, "full_width_output_max_diff": 0 if mode == "neuron_coupled" else 0.1,
             "no_gate_output_max_diff": 0, "evidence_seconds": 0.004,
             "permutation_seconds": 0.0002} for t in (15, 30)]
        log = [{"epoch": i, "validation_accuracy": acc,
                "validation_ce": 1.0 - acc, "width": 40} for i in range(1, 51)]
        row = {"dataset": "Spiral", "seed": seed, "schedule": "fixed",
               "method": method, "gate_mode": mode,
               "final_test_accuracy": acc, "final_test_ce": 1.0 - acc,
               "final_validation_accuracy": acc, "final_validation_ce": 1.0 - acc,
               "optimizer_steps": 50, "training_examples": 6400}
        manifest = {**row, "git_sha": source, "epochs": 50, "fixed_epochs": [15, 30],
                    "max_interventions": 2, "test_examples_used_for_selection": 0,
                    "test_evaluated_after_training": True,
                    "interventions": events, "epochs_log": log, "split": {"seed": seed}}
        folder = root / "Spiral" / f"seed{seed}" / "fixed" / method / mode
        _write(folder / "run_manifest.json", manifest)
        runs.append(row)
    _write(root / "summary.json", {"runs": runs, "failures": []})


def test_locked_grid_contract():
    g = study.grid()
    assert len(g) == study.EXPECTED_RUNS == 45
    assert len(set(g)) == len(g)
    assert sum(m == "identity" for _, m, _ in g) == 5
    assert {(m, g) for _, m, g in g if m != "identity"} == {
        (m, g) for m in ("random", "W", "G", "GI") for g in study.MODES}
    assert study.EVENT_EPOCHS == (15, 30)


def test_plan_is_not_overwritten(tmp_path):
    cfg = study.plan("c" * 40, device="cpu", learning_rate=0.01,
                     save_model=False, max_calibration_batches=None)
    study._verify_or_create_plan(tmp_path, cfg, resume=False)
    assert (tmp_path / "study_plan.json").exists()
    with pytest.raises(ValueError, match="--resume"):
        study._verify_or_create_plan(tmp_path, cfg, resume=False)
    study._verify_or_create_plan(tmp_path, cfg, resume=True)
    with pytest.raises(ValueError, match="mismatch"):
        study._verify_or_create_plan(tmp_path, {**cfg, "source_sha": "d" * 40}, resume=True)


def test_nonempty_unplanned_output_rejected(tmp_path):
    (tmp_path / "unexpected").write_text("keep me")
    cfg = study.plan("c" * 40, device="cpu", learning_rate=0.01,
                     save_model=False, max_calibration_batches=None)
    with pytest.raises(ValueError, match="Non-empty"):
        study._verify_or_create_plan(tmp_path, cfg, resume=False)


def test_exact_pair_randomization_limits():
    s = study._effect([0.01] * 5)
    assert s["mean"] == pytest.approx(0.01)
    assert s["exact_two_sided_sign_flip_p"] == pytest.approx(0.0625)
    assert s["positive"] == 5
    assert study._effect([0.0] * 5)["exact_two_sided_sign_flip_p"] == 1.0


def test_audit_and_analysis(tmp_path):
    _fake_complete_grid(tmp_path)
    audited = study.audit(tmp_path)
    assert audited["pass"] is True
    assert audited["checked_manifests"] == 45
    report = study.analyze(tmp_path)
    assert report["n_runs"] == 45
    assert report["primary_result"]["accuracy_diff_mean"] == pytest.approx(0.035)
    assert report["primary_result"]["accuracy_diff_exact_signflip_p"] == 0.0625
    paths = ["paired_seed_deltas.csv", "paired_contrast_summary.csv",
             "validation_trajectories.csv", "intervention_events.csv"]
    for name in paths:
        assert (tmp_path / "causal_analysis" / name).exists()
    with (tmp_path / "causal_analysis" / "paired_seed_deltas.csv").open() as f:
        assert len(list(csv.DictReader(f))) == len(study.CONTRASTS) * 5
    with (tmp_path / "causal_analysis" / "validation_trajectories.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 45 * 50


def test_wrong_event_epoch_fails(tmp_path):
    _fake_complete_grid(tmp_path)
    path = tmp_path / "Spiral/seed42/fixed/G/positional/run_manifest.json"
    data = json.loads(path.read_text())
    data["interventions"][0]["epoch"] = 16
    _write(path, data)
    with pytest.raises(ValueError, match="event epochs"):
        study.audit(tmp_path)
    assert json.loads((tmp_path / "audit.json").read_text())["pass"] is False


def test_training_budget_mismatch_fails(tmp_path):
    _fake_complete_grid(tmp_path)
    path = tmp_path / "Spiral/seed42/fixed/G/positional/run_manifest.json"
    data = json.loads(path.read_text())
    data["training_examples"] -= 1
    _write(path, data)
    with pytest.raises(ValueError, match="summary/manifest examples"):
        study.audit(tmp_path)


def test_gate_invariance_fails(tmp_path):
    _fake_complete_grid(tmp_path)
    path = tmp_path / "Spiral/seed42/fixed/G/neuron_coupled/run_manifest.json"
    data = json.loads(path.read_text())
    data["interventions"][0]["full_width_output_max_diff"] = 0.01
    _write(path, data)
    with pytest.raises(ValueError, match="coupled full-width equivalence"):
        study.audit(tmp_path)


def test_separation_leakage_fails(tmp_path):
    _fake_complete_grid(tmp_path)
    path = tmp_path / "Spiral/seed42/fixed/G/positional/run_manifest.json"
    data = json.loads(path.read_text())
    data["test_examples_used_for_selection"] = 1
    _write(path, data)
    with pytest.raises(ValueError, match="test_examples_used_for_selection"):
        study.audit(tmp_path)
