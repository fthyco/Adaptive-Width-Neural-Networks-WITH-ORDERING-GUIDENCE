# Baseline audit — engineering provenance

- Authoritative original AWN baseline on main: d55e955bc289a3eefd927c535b14dfd93e2b2dfd.
- Extension base: exp/evidence-sweep-two-phases at 7fd937032d25b15f60f15317be4a59fe379612b6.
- Original AWN files remain unchanged: model.py, distribution.py, dynamic_architecture.py, dynamic_optimizers.py, metric.py.
- Reuse: experiments/bottleneck_triggered/evidence_sweep/controlled_bt.py and identity.py; experiments/bottleneck_triggered/permutation/permutation_manager.py.
- Historic fixed-epoch evidence runs and newer checkpoint counterfactual pilot must be kept in separate cohorts. Their original seeds and test outcomes have been inspected and are not untouched confirmation.
- The original AWN quantile width update and ELBO remain active in skip and intervention continuation.
- Initial benchmark tasks Spiral and SpiralHard are development tasks; DoubleMoon is a saturation sanity check.
- Current runtime support is limited to one-hidden-layer AWN DynamicMLP with exponential gates and single-process deterministic loaders.
- Missing runtime verification: official AWN environment tests, skip-skip on native AWN with width changes, CUDA replay, matched pilot execution. Do not claim G0 success from scaffold tests alone.

## Research gates

G0: native AWN exact fork and skip-skip equivalence.
G1: distinguish positional function changes from neuron-coupled gate preservation and exclude no-ops.
G2 onwards: not implemented in this branch's initial foundation.
