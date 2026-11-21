"""Gate G0: native AWN checkpoint replay, including forced-width stress cases.

Tests the *unchanged* AWN ELBO against the no-intervention AWNAdapter path.
Artificial rate interventions appear ONLY in the explicit width-stress test
and are identical in both forks; they are not research treatment outcomes.

Run (real AWN environment):
  python -m experiments.structural_value.scripts.verify_g0 --output g0.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..actions.action_spec import ActionKind, ActionSpec
from ..adapters.awn_adapter import AWNAdapter
from ..counterfactual.fork_runner import paired_rollouts
from ..state.snapshot import TrainingSnapshot


def _config(n: int) -> dict:
    # A self-contained, real AWN/DynamicMLP setup, not a toy substitute.
    return {
        "num_hidden_layers": 1,
        "num_hidden_neurons": 256,
        "share_width_distribution": False,
        "disable_adaptation": False,
        "n_observations": n,
        "quantile": 0.9,
        "theta_prior_scale": 1.0,
        "alpha_prior_mean": 0.01,
        "alpha_prior_scale": 0.01,
        "dynamic_architecture": {"class_name": "dynamic_architecture.DynamicMLP"},
        "truncated_distribution": {
            "class_name": "distribution.TruncatedDistribution",
            "args": {
                "discretized_distribution": {
                    "class_name": "distribution.DiscretizedDistribution",
                    "args": {"base_distribution": {
                        "class_name": "distribution.Exponential",
                        "args": {"rate": 0.01, "boundary": 0.0005},
                    }},
                },
            },
        },
    }


def _set_test_width(model, target: int) -> None:
    """Force a width change in BOTH arms to stress model/optimizer migration.

    Not part of the prospective action library. Only used for G0 integrity.
    """
    if target < 3:
        raise ValueError("Invalid width stress target")
    dist = model.variational_Ws[0].discretized_distribution.base_distribution
    # Discretized Exponential quantile is int(q)+2 on this reference code.
    # Place continuous quantile safely in the integer bin for 'target'.
    continuous_q = target - 1.5
    rate = -np.log(1.0 - model.quantile) / continuous_q
    value = torch.as_tensor(rate, dtype=dist._rate.dtype,
                            device=dist._rate.device)
    with torch.no_grad():
        dist._rate.copy_(torch.log(torch.expm1(value)))
    observed = model.variational_Ws[0].compute_truncation_number()
    if observed != target:
        raise AssertionError(f"AWN width fixture: {observed} != {target}")


def _step(model, optimizer, elbo, x, y) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = model(x)
    preds, targets = elbo.get_predictions_and_targets(y[:, None], *result)
    loss = elbo.compute_metric(targets, preds)
    if not bool(torch.isfinite(loss)):
        raise AssertionError("Nonfinite native AWN ELBO")
    loss.backward()
    optimizer.step()
    return float(loss.detach().item())


def _observe(model, optimizer, probe) -> dict:
    # Observe exact model/optimizer tensors and distributions after each step.
    named = dict(model.named_parameters())
    optimized = [p for group in optimizer.param_groups for p in group["params"]]
    if len(set(optimized)) != len(optimized):
        raise AssertionError("Duplicate parameter in optimizer groups")
    if set(optimized) != set(named.values()):
        raise AssertionError("AWN optimizer lost/live-parameter alignment")
    if set(model.old_parameters) != set(named):
        raise AssertionError("AWN old-parameter name contract diverged")
    if any(model.old_parameters[k] is not p for k, p in named.items()):
        raise AssertionError("AWN old-parameter identity contract diverged")
    state = {}
    for name, p in named.items():
        state[name] = {
            "parameter": p.detach().cpu().clone(),
            "optimizer": {},
        }
        for key, value in optimizer.state.get(p, {}).items():
            if torch.is_tensor(value):
                if value.numel() != 1 and value.shape != p.shape:
                    raise AssertionError(f"Optimizer state shape mismatch: {name}/{key}")
                state[name]["optimizer"][key] = value.detach().cpu().clone()
            else:
                state[name]["optimizer"][key] = value

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(probe, skip_update=True)[0].detach().cpu().clone()
    finally:
        model.train(was_training)
    width = int(model.dyn_model.get_layer(0).weight.shape[0])
    learned_width = int(model.variational_Ws[0].compute_truncation_number())
    rate = model.variational_Ws[0].discretized_distribution.base_distribution.rate
    return {
        "logits": logits,
        "physical_width": width,
        "learned_width": learned_width,
        "rate": rate.detach().cpu().clone(),
        "parameter_and_optimizer": state,
        "torch_rng": torch.random.get_rng_state().clone(),
    }


def _same(a, b, label: str) -> None:
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
            raise AssertionError(f"Nonidentical G0 tensor: {label}")
        return
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise AssertionError(f"Nonidentical G0 keys: {label}")
        for key in a:
            _same(a[key], b[key], f"{label}.{key}")
        return
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise AssertionError(f"Nonidentical G0 list: {label}")
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{label}[{i}]")
        return
    if a != b:
        raise AssertionError(f"Nonidentical G0 value {label}: {a!r} != {b!r}")


def _run(snapshot, batches, validation, n, *,
         use_adapter: bool, forced_widths: tuple[int, ...] | None) -> list[dict]:
    from metric import ELBO_Classification

    restored = snapshot.fork()
    model, optimizer = restored.model, restored.optimizer
    elbo = ELBO_Classification(n)
    probe = next(iter(validation))[0]
    adapter = AWNAdapter(
        restored, calibration=validation, validation=validation,
        train_examples=n, device=torch.device("cpu"),
        namespace="native-g0-integrity", epoch=2,
    ) if use_adapter else None
    # A true skip must not install an identity/gate override.
    scope = nullcontext()
    trace = []
    with scope:
        if adapter is not None:
            effect = adapter.apply(ActionSpec(ActionKind.SKIP))
            if not effect.no_op or effect.evidence_seconds or effect.intervention_seconds:
                raise AssertionError("Skip must not modify the AWN state or acquire evidence")
        for index, (x, y) in enumerate(batches):
            if forced_widths is not None:
                _set_test_width(model, forced_widths[index])
            if adapter is None:
                loss = _step(model, optimizer, elbo, x, y)
            else:
                # Same native ELBO under the adapter, with AWN's width update.
                adapter.train_step((x, y))
                # Capture the scalar loss on a separate, mutation-free
                # validation input; no extra backward/optimizer step.
                loss = None
            observation = _observe(model, optimizer, probe)
            observation["loss"] = loss
            trace.append(observation)
    return trace


def verify_native_g0(*, seed: int = 42) -> dict:
    from metric import ELBO_Classification
    from model import AWN

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = 48
    model = AWN(2, 2, _config(n))
    model.to(torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.torch_optimizer = optimizer
    generator = torch.Generator().manual_seed(seed + 100)
    x = torch.randn(n, 2, generator=generator)
    y = (x[:, 0] + 0.2 * x[:, 1] > 0).long()
    validation = DataLoader(TensorDataset(x[:16], y[:16]), batch_size=8,
                            shuffle=False, num_workers=0)
    batches = [(xx.clone(), yy.clone()) for xx, yy in
               zip(x[16:40].split(6), y[16:40].split(6))]
    if len(batches) != 4:
        raise AssertionError("Invalid synthetic training stream")
    _set_test_width(model, 10)
    elbo = ELBO_Classification(n)
    for xx, yy in batches[:2]:
        _step(model, optimizer, elbo, xx, yy)
    # Real AWN computes qW_probs with a grad_fn. This tests its actual
    # checkpoint lifecycle, rather than saving an artificially detached model.
    if not any(v.grad_fn is not None for v in model.qW_probs):
        raise AssertionError("Native AWN gate cache did not exercise autograd")
    snapshot = TrainingSnapshot.capture(model, optimizer, provenance={"seed": seed})
    if any(v.grad_fn is None for v in model.qW_probs):
        raise AssertionError("Snapshot capture permanently detached live AWN")
    widths = {}
    for label, forced in (
        ("natural", None),
        ("forced_width_migration", (8, 13, 7, 12)),
    ):
        baseline = _run(snapshot, batches, validation, n,
                        use_adapter=False, forced_widths=forced)
        replay = _run(snapshot, batches, validation, n,
                      use_adapter=True, forced_widths=forced)
        independent_skip = _run(snapshot, batches, validation, n,
                                use_adapter=True, forced_widths=forced)
        for i, (a, b, c) in enumerate(zip(baseline, replay, independent_skip)):
            # A recorded training loss is not available on the adapter path;
            # compare complete state/logits/rate/optimizer/RNG instead.
            a.pop("loss")
            b.pop("loss")
            c.pop("loss")
            _same(a, b, f"{label}.baseline_vs_skip.step{i}")
            _same(b, c, f"{label}.skip_vs_skip.step{i}")
        observed_widths = [r["physical_width"] for r in replay]
        widths[label] = observed_widths
        if forced is not None and len(set(observed_widths)) < 3:
            raise AssertionError("Width-stress scenario did not grow and shrink")
    # Exercise the actual native counterfactual engine at real intermediate
    # horizons. Wall clock is a measurement and is not required to be equal.
    def factory(restored):
        return AWNAdapter(
            restored, calibration=validation, validation=validation,
            train_examples=n, device=torch.device("cpu"),
            namespace="native-g0-fork-runner",
        )

    runner_kwargs = {
        "run_id": f"synthetic:seed{seed}",
        "checkpoint_id": f"synthetic:seed{seed}:postwarmup",
        "rollout_seed": seed + 991,
    }
    first = paired_rollouts(
        snapshot, factory, batches, [ActionSpec(ActionKind.SKIP)],
        [1, 2, 4], **runner_kwargs)
    second = paired_rollouts(
        snapshot, factory, batches, [ActionSpec(ActionKind.SKIP)],
        [1, 2, 4], **runner_kwargs)
    if len(first) != len(second) or len(first) != 3:
        raise AssertionError("Wrong actual-horizon checkpoint counts")
    for left, right in zip(first, second):
        if (left.horizon_steps != right.horizon_steps or
                left.validation_risk != right.validation_risk or
                left.final_widths != right.final_widths or
                left.cost["optimizer_steps"] != right.cost["optimizer_steps"] or
                left.cost["examples"] != right.cost["examples"] or
                not left.no_op or not right.no_op or
                left.immediate_risk_before != left.immediate_risk_after or
                right.immediate_risk_before != right.immediate_risk_after):
            raise AssertionError("Native paired engine skip replay diverged")

    return {
        "schema": "awn_structural_value_g0_v1",
        "seed": seed, "device": "cpu",
        "native_model": "AWN/DynamicMLP",
        "optimizer": "Adam",
        "comparison": "original_AWN_vs_skip_adapter_and_independent_skip",
        "tensor_equality": "exact",
        "steps_per_case": len(batches),
        "paired_engine_horizons": [1, 2, 4],
        "independent_skip_replays": 2,
        "width_trajectories": widths,
        "cases": ["natural", "forced_width_migration"],
        "passed": True,
        "note": "Synthetic examples. Forced width case tests migration only; not a learned intervention.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_native_g0(seed=args.seed)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
