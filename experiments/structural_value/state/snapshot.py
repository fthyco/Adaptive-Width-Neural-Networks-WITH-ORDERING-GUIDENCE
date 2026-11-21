"""Restorable in-memory checkpoint for deterministic AWN research forks.

The model and optimizer are deep-copied *in one object graph*, preserving
parameter references and optimizer state across dynamic-width replacements.
The original AWN qW_probs cache may contain non-leaf autograd tensors, which
cannot be deep-copied by PyTorch. It is a derived forward cache, not a trainable
state; detach it only during copy, then restore the original object unchanged.
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
        py_rng = random.getstate()
        np_rng = copy.deepcopy(np.random.get_state())
        torch_rng = torch.random.get_rng_state().clone()
        cuda_rng = (tuple(state.clone() for state in torch.cuda.get_rng_state_all())
                    if torch.cuda.is_available() else ())
        # AWN.qW_probs is derived on every AWN.forward and often has a grad_fn.
        # PyTorch forbids deepcopy(non-leaf Tensor). Do not detach the real
        # cache permanently, and do not mutate trainable parameters.
        old_cache = getattr(model, "qW_probs", None)
        is_awn_cache = hasattr(model, "qW_probs")
        if is_awn_cache:
            if not isinstance(old_cache, (tuple, list)):
                raise TypeError("Unsupported qW_probs cache shape")
            model.qW_probs = [p.detach().clone() for p in old_cache]
        try:
            bundle = copy.deepcopy(ForkState(model, optimizer, scheduler, scaler,
                                             history, identity, provenance))
        finally:
            if is_awn_cache:
                model.qW_probs = old_cache

        return cls(bundle, py_rng, np_rng, torch_rng, cuda_rng, loader_states)

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
        if hasattr(cloned.model, "torch_optimizer"):
            cloned.model.torch_optimizer = cloned.optimizer
        self.restore_rng(loader)
        return cloned
