"""In-memory full-state forks for one-process deterministic experiments.

The model and optimizer are deep-copied *together*, preserving shared parameter
references even when AWN has dynamically replaced layer parameters. Snapshots
must be taken outside temporary gate/width monkeypatch contexts. Explicit
recorded minibatch streams are the authoritative paired rollout input.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch


def _generators(loader: Any) -> dict[str, torch.Generator]:
    if loader is None:
        return {}
    if getattr(loader, "num_workers", 0) != 0:
        raise ValueError("Exact in-process replay requires num_workers=0")
    candidates = {
        "loader": getattr(loader, "generator", None),
        "sampler": getattr(getattr(loader, "sampler", None), "generator", None),
        "batch_sampler": getattr(
            getattr(getattr(loader, "batch_sampler", None), "sampler", None),
            "generator", None),
    }
    return {key: gen for key, gen in candidates.items()
            if isinstance(gen, torch.Generator)}


@dataclass
class ForkState:
    model: Any
    optimizer: Any
    scheduler: Any = None
    scaler: Any = None
    history: Any = None
    identity: Any = None
    provenance: Any = None


@dataclass(frozen=True)
class TrainingSnapshot:
    """Full restorable state, including CPU/CUDA, Python, NumPy, and loader RNG."""
    state: ForkState
    python_rng: object
    numpy_rng: object
    torch_rng: torch.Tensor
    cuda_rng: tuple[torch.Tensor, ...]
    loader_rng: dict[str, torch.Tensor]

    @classmethod
    def capture(cls, model: Any, optimizer: Any, *, scheduler: Any = None,
                scaler: Any = None, loader: Any = None,
                history: Any = None, identity: Any = None,
                provenance: Any = None) -> "TrainingSnapshot":
        loader_states = {name: gen.get_state().clone()
                         for name, gen in _generators(loader).items()}
        # Deep-copy a single object graph: optimizer.state uses Parameter keys.
        bundle = copy.deepcopy(ForkState(model, optimizer, scheduler, scaler,
                                         history, identity, provenance))
        return cls(bundle, random.getstate(), copy.deepcopy(np.random.get_state()),
                   torch.random.get_rng_state().clone(),
                   tuple(state.clone() for state in torch.cuda.get_rng_state_all())
                   if torch.cuda.is_available() else (),
                   loader_states)

    def restore_rng(self, loader: Any = None) -> None:
        random.setstate(self.python_rng)
        np.random.set_state(copy.deepcopy(self.numpy_rng))
        torch.random.set_rng_state(self.torch_rng.clone())
        if self.cuda_rng:
            if not torch.cuda.is_available() or torch.cuda.device_count() != len(self.cuda_rng):
                raise RuntimeError("CUDA device set differs from checkpoint")
            torch.cuda.set_rng_state_all([s.clone() for s in self.cuda_rng])
        if loader is not None:
            generators = _generators(loader)
            if set(generators) != set(self.loader_rng):
                raise RuntimeError("Loader RNG generator contract changed")
            for name, state in self.loader_rng.items():
                generators[name].set_state(state.clone())

    def fork(self, *, loader: Any = None) -> ForkState:
        cloned = copy.deepcopy(self.state)
        # Some AWN entrypoints store their optimizer on the model; make the
        # relationship explicit even if an upstream deepcopy changes it.
        if hasattr(cloned.model, "torch_optimizer"):
            cloned.model.torch_optimizer = cloned.optimizer
        self.restore_rng(loader)
        return cloned
