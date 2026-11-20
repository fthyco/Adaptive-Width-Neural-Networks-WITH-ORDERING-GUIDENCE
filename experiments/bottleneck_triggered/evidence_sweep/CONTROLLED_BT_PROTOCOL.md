# Controlled AWN BT intervention experiment

**Status:** Real-training pilot implementation; no experimental performance result is claimed until locally executed and audited.

**Branch:** exp/evidence-sweep-two-phases. The original AWN math/core files are unchanged.

## Purpose

The frozen-checkpoint sweep revealed that the rank of a neuron and the AWN gate attached to its physical position are confounded. This experiment measures whether predetermined evidence ordering improves **continued training**, separating positional from counterfactual neuron-coupled gates.

- **positional:** every physical slot continues to receive the original AWN position-based gate. A permutation changes which neuron receives which gate; full-width output may change at intervention.
- **neuron_coupled:** reorder a neuron's gate priority with its stable identity. At intervention the full-width output must remain equivalent (tolerance 1e-4), including optimizer momentum/variance tensors. When width changes subsequently, surviving gate priorities are re-ranked to the current width support, and new identities get later priorities. This is **not** equivalent to the paper's original probabilistic gate interpretation or ELBO, so it is a controlled alternative, not the official AWN extension.

The gate vector is still computed from the original learned width distribution and remains differentiable in its rate. No learned neuron-wise gate parameter or extra KL is introduced.

## Fixed candidates and controls

Methods: identity, random, W, G, J, I, GI, S, S_tie_shuffle.

- W: product of incoming and outgoing L2 weight norms.
- G: mean magnitude of per-example task-CE gradient w.r.t. the gated head input.
- J: mean absolute activation-times-task-gradient at the gated head input.
- I: measured linear-head masking delta cross-entropy, vectorized from true AWN activations.
- GI: mean-percentile fusion of G and I.
- S: actual graphyco 0.2.0 legacy structural perturbation proxy, NOT a causal neuron attribution.
- S_tie_shuffle: keeps S ranks between score groups, shuffles only exact tied groups, with a deterministic seed.
- random: random ordering with a deterministic seed, independent of validation labels.
- identity: untouched AWN model, single positional baseline per dataset/seed/schedule.

**S remains confounded with birth order; this control does not prove independent structural importance.**

## Data and budget controls

The existing safe_loaders derives disjoint train/calibration/validation subsets when the original val/test alias each other. Every run shares the same dataset/seed split. The test partition is accessed only once after training ends and never passed to the detector, scorer, or permutation verification.

Compare matched pairs at identical dataset, seed, schedule, epoch budget and learning rate; record exact training examples, optimizer steps and overhead. For fixed schedules, all selected methods intervene at the same predetermined epoch(s). For endogenous BT triggers, event times can differ after intervention and must be reported, not silently treated as event-matched.

The evaluation is task cross-entropy/accuracy and adapted AWN ELBO during training, not the frozen-prefix score. This initial runner has no preselected checkpoint-based prefix AUC metric after intervention: do not claim one.

### Detector defaults

- monitor: each epoch
- K = 3
- relative improvement threshold = 0.005
- patience = 2
- min epoch = 6
- cooldown = 5 epochs
- max interventions = 2 (configurable)

For a quick controlled smoke check, run the two-epoch or three-epoch small-calibration experiment with a fixed intervention at epoch 1. **This smoke is not evidence of improved AWN.**

## Install and verify

From the repository root:

```bash
python -m pytest -q experiments/bottleneck_triggered/evidence_sweep/tests
python -m experiments.bottleneck_triggered.evidence_sweep.controlled_bt \
  --dataset DoubleMoon --seed 42 --method GI --gate-mode neuron_coupled \
  --schedule fixed --fixed-epochs 1 --epochs 2 \
  --max-interventions 1 --max-calibration-batches 1 \
  --device cpu --output RESULTS/controlled_bt/smoke_gi_coupled
```

Confirm the smoke manifest has exactly one intervention and the recorded full-width max output difference is <= 1e-4. Do not remove this assertion to make the run pass.

## Pilot: matched fixed-event training

The following sequential pilot uses DoubleMoon seeds 42 and 456, fixed intervention epochs 10 and 20, both gate modes, and one unchanged AWN baseline per seed:

```bash
python -m experiments.bottleneck_triggered.evidence_sweep.controlled_bt_suite \
  --datasets DoubleMoon --seeds 42,456 \
  --methods identity,random,W,G,J,I,GI,S,S_tie_shuffle \
  --gate-modes positional,neuron_coupled \
  --schedules fixed --epochs 30 --fixed-epochs 10,20 \
  --max-interventions 2 --device cpu \
  --output RESULTS/controlled_bt/pilot
```

### Second stage: endogenous BT, only after pilot passes

```bash
python -m experiments.bottleneck_triggered.evidence_sweep.controlled_bt_suite \
  --datasets DoubleMoon,Spiral,SpiralHard \
  --seeds 42,123,456,789,1000 \
  --methods identity,random,W,G,J,I,GI,S,S_tie_shuffle \
  --gate-modes positional,neuron_coupled --schedules bt \
  --max-interventions 2 --device cpu --resume \
  --output RESULTS/controlled_bt/endogenous
```

The extended stage runs sequentially and may be expensive. Do not run it before the fixed-event pilot and scientific checks pass.

## Required validation

1. Confirm original AWN remains byte-identical in core modules.
2. Validate full-width output preservation in neuron_coupled immediately after each permutation.
3. Validate no-gate functional equivalence for both modes.
4. Validate Adam state follows each neuron's incoming row, bias and outgoing column.
5. Validate identity and gate priorities across growth/shrink, including intervention after width changes.
6. Validate training steps/examples and exact data split provenance.
7. Check the actual number of fired interventions; some endogenous BT runs can have zero.
8. Avoid selecting winners by test metrics; use validation and an independently decided candidate set.
9. Compare final test accuracy and CE via paired dataset/seed analyses. Report each individual seed and uncertainty, not just maxima.
10. Record the branch HEAD and local changes. No merge to main and no unauthorized push.

## Output contract

Each run writes an individual run_manifest.json containing training logs, optimizer counts, width trajectories, score and permutation evidence, intervention timestamps, output differences, final test accuracy and CE, and resource overhead.

The grid runner incrementally writes summary.csv and summary.json under its output directory, even if later runs fail. These are descriptive paired comparisons, not statistical significance claims.

## Known limitations

- One-hidden-layer AWN MLP; currently CPU-first.
- In the coupled control, truncation/prefix selection after width changes follows current physical slot, while gate *priority* follows surviving identity; this is a deliberately counterfactual model and cannot be described as unchanged AWN.
- S is a structural proxy with many tied scores and possible birth-order confounding.
- G and J are measured at the gated head input; these definitions are not automatically the same as those in previous GRAPHyco experiments.
- I is computed by vectorizing exact head-output removal, not by rerunning a whole model separately for every neuron.
- Independent final test evaluation is a single end-of-training report per run, and must not become part of hyperparameter or candidate selection.
- Running two gate modes does not imply two independent samples; differences should be analyzed as paired outcomes.
