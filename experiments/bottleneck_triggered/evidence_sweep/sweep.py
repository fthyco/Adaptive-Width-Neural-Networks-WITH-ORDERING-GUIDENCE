"""Auditable, frozen-checkpoint AWN evidence sweeps.

Stage 1 enumerates 255 subsets with a single mean-percentile fusion.
Stage 2 independently enumerates the 741 *additional* multi-family/fusion pairs.
Consumes measured, per-neuron evidence; never fabricates telemetry or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FAMILIES = ("W", "S", "A", "G", "J", "E", "H", "I")
ADDITIONAL_FUSIONS = ("robust_z_mean", "geometric_percentile", "pareto_fronts")
FRACTIONS = tuple(round(i / 10, 1) for i in range(1, 11))


def candidate_configs(phase: int) -> list[tuple[tuple[str, ...], str]]:
    if phase not in (1, 2):
        raise ValueError("phase must be 1 or 2")
    subsets = [subset for size in range(1, 9)
               for subset in itertools.combinations(FAMILIES, size)]
    if phase == 1:
        result = [(subset, "mean_percentile") for subset in subsets]
    else:
        result = [(subset, fusion) for subset in subsets if len(subset) > 1
                  for fusion in ADDITIONAL_FUSIONS]
    assert len(result) == (255 if phase == 1 else 741)
    return result


def _as_text(values: np.ndarray) -> list[str]:
    result = [str(x) for x in values.tolist()]
    if len(result) != len(set(result)) or any(not x for x in result):
        raise ValueError("stable_ids must be unique, nonempty strings")
    return result


def _meta(npz: Any) -> dict:
    if "metadata_json" not in npz:
        raise ValueError("metadata_json required for source provenance and validation")
    val = npz["metadata_json"]
    if val.size != 1:
        raise ValueError("metadata_json must be one JSON string")
    parsed = json.loads(str(val.item()))
    if not isinstance(parsed, dict):
        raise ValueError("metadata_json must be an object")
    return parsed


def read_evidence(path: Path) -> tuple[list[str], dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as cache:
        if "stable_ids" not in cache:
            raise ValueError("stable_ids are required; physical indices are not stable IDs")
        ids = _as_text(cache["stable_ids"])
        n = len(ids)
        if n < 1:
            raise ValueError("zero-neuron cache is not meaningful")
        meta = _meta(cache)
        if meta.get("granularity") != "neuron" or meta.get("stage") != "frozen_checkpoint":
            raise ValueError("cache requires granularity=neuron and stage=frozen_checkpoint")
        if "physical_slots" not in cache:
            raise ValueError("physical_slots are required for reproducible AWN permutations")
        slots = np.asarray(cache["physical_slots"])
        if slots.shape != (n,) or not np.array_equal(np.sort(slots), np.arange(n)):
            raise ValueError("initial implementation requires a contiguous active prefix of slots")
        if "active_mask" not in cache or np.asarray(cache["active_mask"]).shape != (n,):
            raise ValueError("active_mask is required")
        if not np.all(cache["active_mask"]):
            raise ValueError("exclude inactive neurons before ranking; keep their IDs in lifecycle registry")
        scores: dict[str, np.ndarray] = {}
        issues: dict[str, str] = {}
        for family in FAMILIES:
            score_key, mask_key = "score_" + family, "valid_" + family
            if score_key not in cache or mask_key not in cache:
                issues[family] = f"missing {score_key} or {mask_key}"
                continue
            score = np.asarray(cache[score_key], dtype=np.float64)
            valid = np.asarray(cache[mask_key])
            if score.shape != (n,) or valid.shape != (n,):
                issues[family] = "invalid score/valid vector shape"
                continue
            if not np.all(valid.astype(bool)) or not np.all(np.isfinite(score)):
                issues[family] = "unobserved or nonfinite neuron evidence; not converted to zero"
                continue
            scores[family] = score
        return ids, scores, {"meta": meta, "issues": issues, "slots": slots.astype(int).tolist()}


def percentiles(values: np.ndarray) -> np.ndarray:
    """Average tied ranks, scaled to [0,1]; constant features contribute 0.5."""
    n = len(values)
    if n == 1 or np.all(values == values[0]):
        return np.full(n, 0.5)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 / (n - 1)
        start = stop
    return ranks


def _robust_z(values: np.ndarray) -> np.ndarray:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale < 1e-12:
        scale = float(np.std(values))
    if scale < 1e-12:
        return np.zeros_like(values)
    return np.clip((values - center) / scale, -5.0, 5.0)


def _pareto(axes: np.ndarray) -> np.ndarray:
    """Nondominated front order, with mean-percentile Borda tie break."""
    n = axes.shape[0]
    dominates = (np.all(axes[:, None, :] >= axes[None, :, :], axis=2)
                 & np.any(axes[:, None, :] > axes[None, :, :], axis=2))
    remaining = np.ones(n, dtype=bool)
    front = np.empty(n, dtype=np.int32)
    layer = 0
    while np.any(remaining):
        active = np.flatnonzero(remaining)
        nondominated = active[~dominates[np.ix_(active, active)].any(axis=0)]
        if not len(nondominated):
            raise RuntimeError("Pareto front extraction failed")
        front[nondominated] = layer
        remaining[nondominated] = False
        layer += 1
    return -front.astype(float) + axes.mean(axis=1) / 2.0


def fuse(families: tuple[str, ...], method: str, scores: dict[str, np.ndarray]) -> np.ndarray:
    missing = [x for x in families if x not in scores]
    if missing:
        raise ValueError(f"unavailable features: {','.join(missing)}")
    if len(families) == 1:
        return percentiles(scores[families[0]])
    axes = np.column_stack([percentiles(scores[f]) for f in families])
    if method == "mean_percentile":
        return axes.mean(axis=1)
    if method == "geometric_percentile":
        return np.exp(np.log(np.clip(axes, 1e-6, 1.0)).mean(axis=1))
    if method == "robust_z_mean":
        return np.column_stack([_robust_z(scores[f]) for f in families]).mean(axis=1)
    if method == "pareto_fronts":
        return _pareto(axes)
    raise ValueError(f"unknown fusion: {method}")


def _softmax_ce(logits: np.ndarray, labels: np.ndarray) -> float:
    safe = logits - np.max(logits, axis=-1, keepdims=True)
    logsumexp = np.log(np.exp(safe).sum(axis=1))
    return float(np.mean(logsumexp - safe[np.arange(len(labels)), labels]))


def _outer(x: np.ndarray, meta: dict) -> np.ndarray:
    activation = meta.get("activation_outer")
    if activation == "none":
        return x
    if activation == "leaky_relu":
        slope = float(meta.get("negative_slope", 0.01))
        return np.where(x >= 0.0, x, x * slope)
    raise ValueError("unsupported outer activation; exporter must supply exact semantics")


def read_frozen_eval(path: Path, stable_ids: list[str]) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        meta = _meta(npz)
        if meta.get("evaluation_role") not in ("validation", "train_calibration_holdout"):
            raise ValueError("screening evaluation must not use test split")
        if _as_text(npz["stable_ids"]) != stable_ids:
            raise ValueError("evaluation and evidence stable IDs differ")
        for key in ("hidden_pre_gate", "outgoing_weight", "output_bias", "position_gates", "labels", "reference_logits"):
            if key not in npz:
                raise ValueError(f"missing eval key {key}")
        hidden = np.asarray(npz["hidden_pre_gate"], dtype=np.float64)
        weight = np.asarray(npz["outgoing_weight"], dtype=np.float64)
        bias = np.asarray(npz["output_bias"], dtype=np.float64)
        gates = np.asarray(npz["position_gates"], dtype=np.float64)
        labels = np.asarray(npz["labels"], dtype=np.int64)
        ref = np.asarray(npz["reference_logits"], dtype=np.float64)
    n, w = hidden.shape
    c = len(bias)
    if (w != len(stable_ids) or weight.shape != (c, w) or gates.shape != (w,)
            or labels.shape != (n,) or ref.shape != (n, c)
            or np.any(labels < 0) or np.any(labels >= c)
            or not all(np.isfinite(v).all() for v in (hidden, weight, bias, gates, ref))):
        raise ValueError("invalid frozen-evaluation tensor dimensions / values")
    actual = _outer(hidden * gates, meta) @ weight.T + bias
    error = float(np.max(np.abs(actual - ref)))
    tol = float(meta.get("reconstruction_atol", 1e-5))
    if error > tol:
        raise ValueError(f"AWN reference reconstruction failed: max_diff={error} tolerance={tol}")
    return {"hidden": hidden, "weight": weight, "bias": bias, "gates": gates,
            "labels": labels, "meta": meta, "identity_error": error}


def evaluate_prefix(order: np.ndarray, cached: dict) -> dict:
    h = _outer(cached["hidden"][:, order] * cached["gates"], cached["meta"])
    w = cached["weight"][:, order]
    contributions = h[:, :, None] * w.T[None, :, :]
    logits_prefix = np.cumsum(contributions, axis=1)
    labels = cached["labels"]
    accuracies, losses = [], []
    for fraction in FRACTIONS:
        k = max(1, min(len(order), int(np.ceil(fraction * len(order)))))
        logits = logits_prefix[:, k - 1, :] + cached["bias"]
        accuracies.append(float(np.mean(np.argmax(logits, axis=1) == labels)))
        losses.append(_softmax_ce(logits, labels))
    return {"prefix_fractions": list(FRACTIONS), "prefix_accuracy": accuracies,
            "prefix_ce": losses, "prefix_accuracy_auc": float(np.trapezoid(accuracies, FRACTIONS))}


def run_phase(phase: int, evidence: Path, output: Path, evaluation: Path | None = None) -> dict:
    ids, scores, audit = read_evidence(evidence)
    eval_data = read_frozen_eval(evaluation, ids) if evaluation is not None else None
    configs = candidate_configs(phase)
    results = []
    for subset, fusion in configs:
        candidate_id = "E" + "".join(subset) + "__" + fusion
        missing = [f for f in subset if f not in scores]
        if missing:
            results.append({"id": candidate_id, "families": list(subset), "fusion": fusion,
                            "status": "skipped_unobserved", "missing_families": missing})
            continue
        ranks = fuse(subset, fusion, scores)
        ordering = np.array(sorted(range(len(ids)), key=lambda i: (-float(ranks[i]), ids[i])), dtype=int)
        out = {"id": candidate_id, "families": list(subset), "fusion": fusion,
               "status": "ranked", "stable_id_order": [ids[i] for i in ordering],
               "physical_slot_order": [audit["slots"][i] for i in ordering],
               "score_spread": float(np.ptp(ranks)),
               "tie_fraction": float(1.0 - len(np.unique(ranks)) / len(ranks))}
        if eval_data is not None:
            out["frozen_checkpoint_evaluation"] = evaluate_prefix(ordering, eval_data)
        results.append(out)
    done = sum(r["status"] == "ranked" for r in results)
    report = {"schema_version": "two_phase_sweep_v1", "phase": phase,
              "phase_name": "255_mean_percentile" if phase == 1 else "741_additional_fusions",
              "source_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
              "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest() if evaluation else None,
              "features_available": list(scores), "source_metadata": audit["meta"],
              "feature_issues": audit["issues"], "planned": len(configs),
              "ranked": done, "skipped": len(configs) - done,
              "evaluated": done if eval_data is not None else 0,
              "evaluation_note": "Frozen-checkpoint screening only; not training or final test accuracy",
              "results": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=(1, 2))
    parser.add_argument("--evidence", type=Path, required=True, help="Measured feature cache (.npz)")
    parser.add_argument("--evaluation", type=Path, help="Optional independent validation cache (.npz)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true", help="Nonzero exit if any candidate lacks evidence")
    args = parser.parse_args(argv)
    report = run_phase(args.phase, args.evidence, args.output, args.evaluation)
    print(json.dumps({key: report[key] for key in ("phase", "planned", "ranked", "skipped", "evaluated")}, indent=2))
    return int(args.require_complete and report["skipped"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
