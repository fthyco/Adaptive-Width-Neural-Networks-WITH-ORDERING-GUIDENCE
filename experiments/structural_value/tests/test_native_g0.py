"""Mandatory native AWN Gate G0; no toy substitutes or pytest skips."""
from experiments.structural_value.scripts.verify_g0 import verify_native_g0


def test_native_awn_checkpoint_skip_equivalence_with_resize_stress():
    report = verify_native_g0(seed=42)
    assert report["passed"] is True
    assert len(report["width_trajectories"]["forced_width_migration"]) == 4
    assert len(set(report["width_trajectories"]["forced_width_migration"])) >= 3
