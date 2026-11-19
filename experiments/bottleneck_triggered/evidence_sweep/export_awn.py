"""Collect neuron-level evidence and frozen AWN MLP validation tensors.

Scope: corrected one-hidden-layer AWN DynamicMLP, classifier with a linear head.
Model must already be trained and physically stable. No optimizer steps are taken.
GRAPHyco supplies the static graph primitives; dynamic neuron signals are
explicitly derived from AWN tensors, NOT inferred from Graphyco FX operation nodes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .sweep import read_frozen_eval


def _check_model(model: torch.nn.Module, stable_ids: Sequence[str]) -> tuple:
    if not hasattr(model, "dyn_model") or model.dyn_model.num_layers != 1:
        raise ValueError("Only one-hidden-layer AWN DynamicMLP currently supported")
    dyn = model.dyn_model
    if (dyn.disable_adaptation or dyn.apply_ntk_reparam
            or dyn.apply_experimental_rescaling):
        raise ValueError("Unsupported AWN gate/rescaling configuration")
    layer0, layer1 = dyn.get_layer(0), dyn.get_layer(1)
    if not isinstance(layer0, torch.nn.Linear) or not isinstance(layer1, torch.nn.Linear):
        raise ValueError("Hidden and output layers must be nn.Linear")
    w = int(layer0.weight.shape[0])
    if len(stable_ids) != w or layer1.in_features != w:
        raise ValueError("Stable IDs must correspond to the complete active physical hidden layer")
    if len(set(stable_ids)) != len(stable_ids):
        raise ValueError("Duplicate stable neuron identities")
    outer = dyn.activation_outer
    if outer is None:
        outer_mode = "none"
    elif outer is F.leaky_relu or getattr(outer, "__name__", "") == "leaky_relu":
        outer_mode = "leaky_relu"
    else:
        raise ValueError("Unsupported activation_outer; exact reference reconstruction required")
    return layer0, layer1, outer_mode


def _static_graph_scores(w_in: np.ndarray, w_out: np.ndarray, threshold: float) -> np.ndarray:
    from graphyco.core.primitives import Node, Edge, GraphState
    from graphyco.core.evaluation import perturbation_profile

    width, input_dim = w_in.shape
    output_dim = w_out.shape[0]
    nodes = {}
    for prefix, count in (("IN", input_dim), ("H", width), ("OUT", output_dim)):
        for j in range(count):
            key = f"{prefix}_{j}"
            nodes[key] = Node(id=key, name=key, purpose=prefix, active=True)
    edges = []
    for weight, src_prefix, dst_prefix in ((w_in, "IN", "H"), (w_out, "H", "OUT")):
        maximum = float(np.abs(weight).max(initial=0.0))
        cutoff = threshold * maximum
        for target, source in np.argwhere(np.abs(weight) > max(cutoff, 0.0)):
            edges.append(Edge(source=f"{src_prefix}_{int(source)}", target=f"{dst_prefix}_{int(target)}"))
    graph = GraphState(nodes=nodes, edges=edges)
    profile = perturbation_profile(graph)
    return np.array([float(profile[f"H_{i}"]) for i in range(width)])


def _attested_history_score(current_ids: Sequence[str], history_paths: Sequence[Path], current_j: np.ndarray) -> np.ndarray:
    if len(history_paths) < 2:
        raise ValueError("Temporal H requires at least two previous chronological checkpoints")
    seq = []
    for path in history_paths:
        with np.load(path, allow_pickle=False) as old:
            if list(map(str, old["stable_ids"].tolist())) != list(current_ids):
                raise ValueError("Temporal history IDs/order changed; supply a stable-ID-aware alignment")
            if not np.all(old["valid_J"]):
                raise ValueError("Missing/invalid prior J evidence must not be zero-filled")
            arr = np.asarray(old["score_J"], dtype=np.float64)
            if arr.shape != current_j.shape or not np.isfinite(arr).all():
                raise ValueError("Invalid J history")
            seq.append(arr)
    series = np.stack([*seq, current_j], axis=0)
    mu = series.mean(axis=0)
    cv = series.std(axis=0) / (np.abs(mu) + 1e-8)
    return mu / (1.0 + cv)


def export_checkpoint(
    model: torch.nn.Module,
    calibration_loader: Iterable,
    validation_loader: Iterable,
    stable_ids: Sequence[str],
    evidence_path: Path,
    evaluation_path: Path,
    *,
    checkpoint_id: str,
    threshold: float = 0.05,
    prior_evidence_paths: Sequence[Path] = (),
) -> None:
    """Export 7 measured/proxy families and H if >=2 prior aligned checkpoints.

    `I` is exact single-neuron masking at the final linear layer on calibration
    examples. `E` is weighted edge-contribution sensitivity, not per-edge FX grad.
    A/G/J/E/I use task cross-entropy; keep the ELBO for actual training.
    Call only on training-calibration and separate validation data, never test.
    """
    if not 0 <= threshold < 1:
        raise ValueError("threshold must be in [0,1)")
    stable_ids = list(stable_ids)
    layer0, layer1, outer_mode = _check_model(model, stable_ids)
    was_training = bool(model.training)
    device = next(model.parameters()).device
    width = len(stable_ids)
    n_classes = int(layer1.out_features)
    evidence = {name: np.zeros(width, dtype=np.float64) for name in ("A", "G", "J", "E", "I")}
    samples = 0
    eval_hidden, eval_labels, eval_logits = [], [], []
    gate_vector = None
    captured: dict = {}

    def capture_head_input(_module, args):
        captured["h"] = args[0]

    hook = layer1.register_forward_pre_hook(capture_head_input)
    try:
        model.eval()
        for x, y in calibration_loader:
            x, y = x.to(device), y.to(device).reshape(-1).long()
            b = int(len(y))
            captured.clear()
            with torch.enable_grad():
                outputs = model(x, skip_update=True)
                logits = outputs[0]
                if "h" not in captured:
                    raise RuntimeError("No AWN head-input tensor captured")
                h = captured["h"]
                if h.shape != (b, width) or logits.shape != (b, n_classes):
                    raise ValueError("Unexpected hidden/logit shapes")
                g = torch.autograd.grad(F.cross_entropy(logits, y), h)[0] * b
                probabilities = torch.softmax(logits.detach(), dim=-1)
                grad_logits = probabilities - F.one_hot(y, num_classes=n_classes)
                # Edge-level first-order contribution magnitude: per-output-edge
                # h_j * W_out[k,j] * dCE/dlogit_k. NOT GRAPHyco FX edge gradients.
                edge_contributions = (h.detach().unsqueeze(-1)
                                      * layer1.weight.detach().T.unsqueeze(0)
                                      * grad_logits.unsqueeze(1)).abs().sum(dim=-1)
                ablated = logits.detach().unsqueeze(1) - (
                    h.detach().unsqueeze(-1) * layer1.weight.detach().T.unsqueeze(0))
                removed_ce = F.cross_entropy(
                    ablated.permute(0, 2, 1), y[:, None].expand(b, width), reduction="none")
                base_ce = F.cross_entropy(logits.detach(), y, reduction="none")
                batch_scores = {
                    "A": h.detach().abs(),
                    "G": g.detach().abs(),
                    "J": (h.detach() * g.detach()).abs(),
                    "E": edge_contributions,
                    "I": removed_ce - base_ce[:, None],
                }
                for key, value in batch_scores.items():
                    evidence[key] += value.double().sum(dim=0).cpu().numpy()
                samples += b
        if samples == 0:
            raise ValueError("calibration_loader had zero examples")
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device).reshape(-1).long()
            with torch.no_grad():
                logits = model(x, skip_update=True)[0]
                probs = model.qW_probs[0].detach()
                if len(probs) != width:
                    raise ValueError("AWN qW vector and active width disagree")
                gate_vector = probs.cpu().numpy().astype(np.float64)
                hidden = model.dyn_model.activation(layer0(x))
                eval_hidden.append(hidden.detach().cpu().numpy())
                eval_labels.append(y.detach().cpu().numpy())
                eval_logits.append(logits.detach().cpu().numpy())
        if not eval_hidden:
            raise ValueError("validation_loader had zero examples")
    finally:
        hook.remove()
        model.train(was_training)

    for k in evidence:
        evidence[k] /= samples
    w_in = layer0.weight.detach().cpu().double().numpy()
    w_out = layer1.weight.detach().cpu().double().numpy()
    evidence["W"] = np.linalg.norm(w_in, axis=1) * np.linalg.norm(w_out, axis=0)
    evidence["S"] = _static_graph_scores(w_in, w_out, threshold)
    if prior_evidence_paths:
        evidence["H"] = _attested_history_score(stable_ids, prior_evidence_paths, evidence["J"])
    feature_meta = {
        "granularity": "neuron", "stage": "frozen_checkpoint", "checkpoint_id": checkpoint_id,
        "calibration_examples": samples, "model_scope": "AWN one-hidden DynamicMLP",
        "AWN_core_modified": False, "source": "AWN tensors + GRAPHyco structural primitives",
        "family_semantics": {
            "W": "incoming_outgoing_weight_l2_product",
            "S": "graphyco_legacy_perturbation_profile_structural_proxy",
            "A": "mean_abs_gated_hidden_activation",
            "G": "mean_abs_per_example_task_gradient_wrt_head_input",
            "J": "mean_abs_paired_activation_gradient_taylor",
            "E": "derived_absolute_output_edge_contribution_taylor_NOT_measured_FX_edge_grad",
            "H": "mean_taylor_over_3_or_more_real_checkpoints_divided_by_1_plus_CV",
            "I": "exact_single_neuron_head_masking_delta_CE_on_train_calibration",
        },
        "graph_threshold_relative_to_layer_max": threshold,
        "history_checkpoint_count": len(prior_evidence_paths) + 1 if prior_evidence_paths else 0,
        "notes": "CE used only for attribution/ablation; no parameter update; validation reserved for screening",
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evidence_path,
                        stable_ids=np.asarray(stable_ids, dtype=str),
                        physical_slots=np.arange(width, dtype=np.int64),
                        active_mask=np.ones(width, dtype=bool),
                        metadata_json=json.dumps(feature_meta),
                        **{"score_" + k: v for k, v in evidence.items()},
                        **{"valid_" + k: np.isfinite(v) for k, v in evidence.items()})
    bias = (layer1.bias.detach().cpu().numpy()
            if layer1.bias is not None else np.zeros(n_classes, dtype=np.float64))
    meta_eval = {"evaluation_role": "validation", "activation_outer": outer_mode,
                 "negative_slope": 0.01, "checkpoint_id": checkpoint_id,
                 "reconstruction_atol": 2e-5}
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evaluation_path,
                        stable_ids=np.asarray(stable_ids, dtype=str),
                        hidden_pre_gate=np.concatenate(eval_hidden),
                        outgoing_weight=w_out,
                        output_bias=bias,
                        position_gates=gate_vector,
                        labels=np.concatenate(eval_labels),
                        reference_logits=np.concatenate(eval_logits),
                        metadata_json=json.dumps(meta_eval))
    read_frozen_eval(evaluation_path, stable_ids)
