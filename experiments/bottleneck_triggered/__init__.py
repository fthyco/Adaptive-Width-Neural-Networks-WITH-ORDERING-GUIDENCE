"""Bottleneck-triggered research extensions.

Do not import the full AWN training stack when importing independent evidence
utilities. Preserve the previous run_single_experiment package attribute API.
"""

__all__ = ["run_single_experiment"]


def __getattr__(name: str):
    if name == "run_single_experiment":
        from .runner import run_single_experiment
        return run_single_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
