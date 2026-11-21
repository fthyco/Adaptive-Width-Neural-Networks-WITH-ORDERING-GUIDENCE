"""Predeclared 96-fork engineering design. Dependency-free, fail-closed."""
DATASETS = ("Spiral", "SpiralHard")
SEEDS = (42, 123, 456)
CHECKPOINTS = (5, 10, 20, 35)
HORIZON_STEPS = 8
HORIZONS = (1, 4, 8)
ACTIONS = ("skip", "random_positional", "weight_positional", "gradient_positional")
