"""Frozen fresh-seed same-checkpoint G5A exploratory ablation contract.

A G2 historical replay diverged on SpiralHard and is never paired to
different G2 teacher labels. G5A instead collects before-action features
and future outcomes from the SAME saved native AWN checkpoint.
"""
DATASETS = ("Spiral", "SpiralHard")
SEEDS = (20331, 20332, 20333)
CHECKPOINTS = (5, 10, 20, 35)
HORIZONS = (1, 4, 8)
HORIZON_STEPS = 8
ACTIONS = ("skip", "random_positional", "weight_positional", "gradient_positional")
COST_WEIGHT = 0.01
REFERENCE_SECONDS = 1.0
