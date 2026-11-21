"""Mandatory native AWN Gate G0; no toy substitutes or pytest skips."""
import pytest
from experiments.structural_value.scripts.verify_g0 import verify_native_g0


@pytest.mark.parametrize("seed", [42, 123])
def test_native_awn_checkpoint_skip_equivalence_with_resize_stress(seed):
    report = verify_native_g0(seed=seed)
    assert report["passed"] is True
    assert len(report["width_trajectories"]["forced_width_migration"]) == 4
    assert len(set(report["width_trajectories"]["forced_width_migration"])) >= 3
    assert report["paired_engine_horizons"] == [1, 2, 4]
