"""G1: Native AWN action mechanics and matched causal pilot (synthetic data).

Check immediate positional effects versus identity-coupled invariance, physical
permutation, complete Adam moment migration, width-rate continuity, no-op
classification and real paired checkpoint rollouts. Does not claim benefit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..actions.action_spec import ActionKind, ActionSpec
from ..adapters.awn_adapter import AWNAdapter
from ..counterfactual.fork_runner import paired_rollouts
from ..counterfactual.label_builder import build_labels
from ..state.snapshot import TrainingSnapshot
from .verify_g0 import _config, _observe, _set_test_width, _step


def _logits(model, x):
    mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            return model(x, skip_update=True)[0].detach().cpu().clone()
    finally:
        model.train(mode)


def _named_state(model, optimizer):
    """Observe parameters, Adam buffers, and ensure dynamic optimizer alignment."""
    named = dict(model.named_parameters())
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len(optimized) != len(set(optimized)) or set(optimized) != set(named.values()):
        raise AssertionError("Optimizer/parameter references not a bijection")
    if set(model.old_parameters) != set(named) or any(
        model.old_parameters[k] is not p for k, p in named.items()
    ):
        raise AssertionError("AWN dynamic old_parameters pointers are stale")
    values = {}
    for name, p in named.items():
        state = {}
        for key, val in optimizer.state.get(p, {}).items():
            state[key] = val.detach().clone() if torch.is_tensor(val) else val
        values[name] = (p.detach().clone(), state)
    return values


def _equal_value(a, b, where):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or not torch.equal(a, b):
            raise AssertionError(f"Adam/permutation mismatch: {where}")
    elif a != b:
        raise AssertionError(f"Adam/non-tensor state mismatch: {where}")


def _check_permuted_state(before, after, order):
    """All params/states obey the same neuron map, no random Adam reinit."""
    if before.keys() != after.keys():
        raise AssertionError("Parameter names changed during permutation")
    for name in before:
        incoming, old_state = before[name]
        outgoing, new_state = after[name]
        if name.endswith("dyn_model.layers.0.weight") or name == "dyn_model.layers.0.weight":
            expected = incoming[order, :]
            transform = lambda t: t[order, :]
        elif name.endswith("dyn_model.layers.0.bias") or name == "dyn_model.layers.0.bias":
            expected = incoming[order]
            transform = lambda t: t[order]
        elif name.endswith("dyn_model.layers.1.weight") or name == "dyn_model.layers.1.weight":
            expected = incoming[:, order]
            transform = lambda t: t[:, order]
        else:
            expected = incoming
            transform = lambda t: t
        _equal_value(expected, outgoing, name)
        if old_state.keys() != new_state.keys():
            raise AssertionError(f"Optimizer state keys changed: {name}")
        for key, value in old_state.items():
            # Scalar 'step' is global to the parameter and cannot be permuted.
            expected_state = (transform(value) if torch.is_tensor(value)
                              and value.shape == incoming.shape else value)
            _equal_value(expected_state, new_state[key], f"{name}.{key}")


def _fixture(seed):
    from metric import ELBO_Classification
    from model import AWN

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 139)
    x = torch.randn(96, 2, generator=generator)
    y = ((x[:, 0] * 0.9 - x[:, 1] * 0.3) > 0).long()
    train_x, train_y = x[:64], y[:64]
    cal = DataLoader(TensorDataset(x[64:80], y[64:80]), batch_size=8,
                     shuffle=False, num_workers=0)
    val = DataLoader(TensorDataset(x[80:], y[80:]), batch_size=8,
                     shuffle=False, num_workers=0)
    model = AWN(2, 2, _config(n=64))
    model.to(torch.device("cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = opt
    _set_test_width(model, 12)
    elbo = ELBO_Classification(64)
    for x_b, y_b in zip(train_x[:24].split(6), train_y[:24].split(6)):
        _step(model, opt, elbo, x_b, y_b)
    snapshot = TrainingSnapshot.capture(
        model, opt, provenance={"synthetic": True, "seed": seed})
    continuation = tuple((a.detach().clone(), b.detach().clone())
                         for a, b in zip(train_x[24:48].split(6),
                                         train_y[24:48].split(6)))
    return snapshot, cal, val, continuation


def _adapter(restored, cal, val, seed):
    return AWNAdapter(restored, calibration=cal, validation=val,
                      train_examples=64, device=torch.device("cpu"),
                      namespace=f"native-g1:seed{seed}", epoch=4)


def verify_native_g1(seed=42):
    snapshot, calibration, validation, batches = _fixture(seed)
    probe = next(iter(validation))[0]
    actions = [
        ActionSpec(ActionKind.RANDOM_POS, action_seed=seed + 501),
        ActionSpec(ActionKind.WEIGHT_POS),
        ActionSpec(ActionKind.GRADIENT_POS),
    ]
    result = []
    for action in actions:
        for mode in ("positional", "neuron_coupled"):
            variant = ActionSpec(action.kind, gate_mode=mode,
                                 action_seed=action.action_seed)
            restored = snapshot.fork()
            adapter = _adapter(restored, calibration, validation, seed)
            with adapter.scope(mode):
                model, optimizer = adapter.model, adapter.optimizer
                before = _named_state(model, optimizer)
                logits_before = _logits(model, probe)
                rate_before = (model.variational_Ws[0].discretized_distribution
                               .base_distribution._rate.detach().clone())
                width_before = model.dyn_model.get_layer(0).weight.shape[0]
                effect = adapter.apply(variant)
                after = _named_state(model, optimizer)
                logits_after = _logits(model, probe)
                width_after = model.dyn_model.get_layer(0).weight.shape[0]
                order = effect.details["permutation"] if effect.details else None
                if order is None or sorted(order) != list(range(width_before)):
                    raise AssertionError("Action did not produce bijective provenance")
                if effect.no_op != (order == list(range(width_before))):
                    raise AssertionError("Incorrect no-op classification")
                if width_before != width_after or width_before != adapter.widths()[0]:
                    raise AssertionError("Ordering action changed AWN width")
                if not torch.equal(rate_before,
                                   model.variational_Ws[0].discretized_distribution
                                   .base_distribution._rate):
                    raise AssertionError("Ordering action modified the width-rate parameter")
                _check_permuted_state(before, after, order)
                # AWN full-width dynamic gates stay with their original
                # identities in coupled mode, but follow slots in positional.
                diff = float((logits_after - logits_before).abs().max())
                if mode == "neuron_coupled" and diff > 1e-4:
                    raise AssertionError(f"Identity-coupled output drift: {diff}")
                # Keep this a genuine training-compatibility test, not only
                # a static test of the permutation.
                adapter.train_step(batches[0])
                _named_state(model, optimizer)
                if not np.isfinite(adapter.validation_risk()):
                    raise AssertionError("Nonfinite post-action training validation CE")
                result.append({
                    "kind": action.kind.value, "gate_mode": mode,
                    "no_op": effect.no_op,
                    "immediate_logit_max_diff": diff,
                    "permutation_size": len(order),
                    "post_action_train_step": "passed",
                    "optimizer_migration": "exact",
                })

    positional_random = next(r for r in result
                             if r["kind"] == ActionKind.RANDOM_POS.value
                             and r["gate_mode"] == "positional")
    if positional_random["no_op"] or positional_random["immediate_logit_max_diff"] <= 1e-6:
        raise AssertionError("Random positional control did not modify live AWN output")

    # No-op through the same adapter path: score computation is allowed but
    # identity ordering must neither shuffle weights nor migrate Adam buffers.
    restored = snapshot.fork()
    adapter = _adapter(restored, calibration, validation, seed)
    with adapter.scope("positional"):
        before = _named_state(adapter.model, adapter.optimizer)
        width = adapter.model.dyn_model.get_layer(0).weight.shape[0]
        with patch(
            "experiments.bottleneck_triggered.evidence_sweep.controlled_bt.candidate_order",
            return_value=list(range(width)),
        ):
            noop = adapter.apply(ActionSpec(ActionKind.WEIGHT_POS))
        after = _named_state(adapter.model, adapter.optimizer)
        if not noop.no_op or noop.details["permutation"] != list(range(width)):
            raise AssertionError("Identity action not marked as no-op")
        _check_permuted_state(before, after, list(range(width)))

    paired_actions = [ActionSpec(ActionKind.SKIP)] + [
        ActionSpec(action.kind, gate_mode=mode, action_seed=action.action_seed)
        for action in actions for mode in ("positional", "neuron_coupled")
    ]
    def factory(restored):
        return _adapter(restored, calibration, validation, seed)

    paired = paired_rollouts(
        snapshot, factory, batches, paired_actions, [1, 2, 4],
        run_id=f"native-g1:seed{seed}", checkpoint_id=f"seed{seed}:synthetic-cp4",
        rollout_seed=seed + 2009)
    labels = build_labels(paired, cost_weight=0.0, reference_seconds=1.0)
    if len(paired) != len(paired_actions) * 3 or len(labels) != len(paired):
        raise AssertionError("Paired action grid incomplete")
    for horizon in [1, 2, 4]:
        group = [row for row in paired if row.horizon_steps == horizon]
        skip = next(row for row in group if row.action.kind is ActionKind.SKIP)
        for row in group:
            if row.cost["optimizer_steps"] != horizon or row.cost["examples"] != horizon * 6:
                raise AssertionError("Training budget mismatch")
            label = next(l for l in labels if l.action_id == row.action.action_id
                         and l.pair_key == row.pair_key)
            if abs(label.delta_risk - (skip.validation_risk - row.validation_risk)) > 1e-9:
                raise AssertionError("Paired label sign incorrect")
            if label.no_op != row.no_op:
                raise AssertionError("No-op lost in label serialization")
        # Coupled permutations are re-labellings when the effective gate is
        # neuron-coupled; future risk should match skip up to roundoff.
        for row in group:
            if row.action.gate_mode == "neuron_coupled" and not row.no_op:
                if abs(row.validation_risk - skip.validation_risk) > 5e-4:
                    raise AssertionError("Coupled continuation unexpectedly changed risk")

    return {
        "schema": "awn_structural_value_g1_v1",
        "seed": seed, "device": "cpu",
        "test_examples_used": 0,
        "native_actions": result,
        "identity_noop": "verified",
        "paired_horizons": [1, 2, 4],
        "paired_action_count": len(paired_actions),
        "paired_observations": len(paired),
        "labels": "matched_sign_and_budget_verified",
        "validation": "synthetic_disjoint_calibration_validation",
        "passed": True,
        "interpretation": "Engineering controls only; no demonstrated benefit or transfer",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_native_g1(args.seed)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
