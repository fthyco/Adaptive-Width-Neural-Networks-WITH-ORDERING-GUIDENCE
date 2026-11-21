"""Pre-intervention descriptors of actual W, G and deterministic random proposals.

Pure feature arithmetic.  No loss label, post-action weights, or future
training minibatches enter these descriptors.  The candidate is measured,
not applied.  These are positional-gate proposal features, not a new AWN gate.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "random_mean_rank_displacement", "random_prefix_replacement",
    "weight_mean_rank_displacement", "weight_prefix_replacement",
    "weight_score_concentration",
    "gradient_mean_rank_displacement", "gradient_prefix_replacement",
    "gradient_score_concentration",
)


def order_features(order, *, n: int, learned_width: int) -> tuple[float, float]:
    """Normalized full-rank displacement and fraction of learned prefix exchanged."""
    if n < 1 or learned_width < 1 or learned_width > n:
        raise ValueError("Invalid AWN width/prefix")
    p = np.asarray(order)
    if p.shape != (n,) or p.dtype.kind not in "iu":
        raise ValueError("Proposal must be an integer permutation")
    if sorted(p.tolist()) != list(range(n)):
        raise ValueError("Proposal is not bijective")
    inverse = np.empty(n, dtype=np.int64)
    inverse[p] = np.arange(n)
    displacement = (float(np.mean(np.abs(inverse - np.arange(n)))) / max(n - 1, 1))
    prefix_replacement = sum(int(j >= learned_width) for j in p[:learned_width]) / learned_width
    return displacement, prefix_replacement


def evidence_order_and_concentration(scores) -> tuple[list[int], float]:
    """Descending evidence; stable physical-index ties (matching fresh tracker IDs)."""
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or x.size < 1 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Evidence must be a finite nonnegative per-neuron vector")
    n = x.size
    order = sorted(range(n), key=lambda j: (-float(x[j]), j))
    mass = float(x.sum())
    k = max(1, (n + 3) // 4)
    concentration = float(np.sum(x[order[:k]]) / mass) if mass > 0 else k / n
    return order, concentration


def proposal_features(*, w_scores, g_scores, learned_width: int,
                      random_action_seed: int) -> dict[str, float]:
    w = np.asarray(w_scores, dtype=np.float64)
    g = np.asarray(g_scores, dtype=np.float64)
    if w.shape != g.shape or w.ndim != 1:
        raise ValueError("W/G evidence widths must match")
    n = len(w)
    if not isinstance(random_action_seed, (int, np.integer)) or random_action_seed < 0:
        raise ValueError("Random action seed must be fixed before intervention")
    random_order = np.random.default_rng(int(random_action_seed)).permutation(n).tolist()
    weight_order, wc = evidence_order_and_concentration(w)
    grad_order, gc = evidence_order_and_concentration(g)
    r_rank, r_prefix = order_features(random_order, n=n, learned_width=learned_width)
    w_rank, w_prefix = order_features(weight_order, n=n, learned_width=learned_width)
    g_rank, g_prefix = order_features(grad_order, n=n, learned_width=learned_width)
    return dict(zip(FEATURE_NAMES, (
        r_rank, r_prefix, w_rank, w_prefix, wc, g_rank, g_prefix, gc,
    )))
