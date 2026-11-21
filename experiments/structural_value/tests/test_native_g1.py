"""Native AWN G1 action integrity, no-op, and matched rollout labels."""
import pytest
from experiments.structural_value.scripts.verify_g1 import verify_native_g1


@pytest.mark.parametrize("seed", [42, 123])
def test_native_awn_actions_and_matched_labels(seed):
    report = verify_native_g1(seed)
    assert report["passed"]
    assert report["identity_noop"] == "verified"
    assert report["paired_action_count"] == 7
    assert report["paired_observations"] == 21
    assert len(report["native_actions"]) == 6
    assert len(report["coupled_future_risk_differences"]) >= 3
    assert all(r["optimizer_migration"] == "exact"
               for r in report["native_actions"])
    assert all(r["immediate_logit_max_diff"] < 1e-4
               for r in report["native_actions"]
               if r["gate_mode"] == "neuron_coupled")
