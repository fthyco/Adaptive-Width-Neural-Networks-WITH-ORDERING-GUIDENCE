# Spiral fixed-event causal comparison (replication v1)

**Scope:** Single-hidden-layer AWN, real training, fixed interventions after epochs 15 and 30. Same branch `exp/evidence-sweep-two-phases`; AWN core is unchanged.

## Research question

On Spiral, prior endogenous BT runs showed larger test accuracy gains under positional gates, but they did not use fixed event times. Does evidence-guided reordering (especially G) outperform a random reordering control with identical intervention timing, and how much of the difference depends on gate reassignment?

**Predeclared primary contrast:** `G / positional` minus `random / positional`, in final test accuracy, paired within seed. The previously observed +4.62 pp versus AWN is context, not the primary contrast. **Secondary** comparisons: G versus W; G versus GI; G versus random under neuron-coupled gates; within-method positional minus neuron-coupled; G and random versus untouched AWN. Report final test CE and validation values for all contrasts.

## Locked design

- Dataset: Spiral only; `safe_loaders` disjoint train/calibration/validation/test rules are reused.
- Seeds: 42, 123, 456, 789, 1000.
- Model: existing AWN baseline, unchanged width distribution, optimizer, LR (0.01), batch definitions and train split provenance.
- Epochs: 50 per run.
- Event times: exactly after training epochs 15 and 30, with no endogenous plateau trigger.
- Methods: untouched `identity`, `random`, `W`, `G`, and `GI`. Identity baseline has no interventions. All other methods use both `positional` and `neuron_coupled` gating.
- Runs: 5 × (1 + 4 × 2) = **45**, sequential on CPU. One AWN baseline per seed, not one per gate mode.
- Validation: used for training monitoring and recording trajectories; not used to change the fixed timing or choose methods during this study.
- Test: assessed at run end only. Metrics are accuracy and CE, not frozen-checkpoint Prefix AUC. Preserve all failures.
- Counterfactual: neuron-coupled gating is *not* the paper's original AWN probabilistic semantics. Verify full-width equivalence immediately after permutation for coupled gates; don't demand it for positional gates.

**Independence limitation:** Seeds and dataset/test partition have been inspected in earlier BT reports. This is a controlled replication/mechanistic follow-up, **not a genuinely new confirmatory holdout**. Do not label nominal p-values as fresh confirmatory evidence or claim a universally superior ranking method.

## Execution

Run commands from the repository root in a verified, clean worktree at the exact new branch commit:

```bash
python -m pytest -q experiments/bottleneck_triggered/evidence_sweep/tests

python -m experiments.bottleneck_triggered.evidence_sweep.spiral_fixed_causal plan

python -m experiments.bottleneck_triggered.evidence_sweep.spiral_fixed_causal run \
  --device cpu --save-model \
  --output RESULTS/controlled_bt/spiral_fixed_causal_v1

python -m experiments.bottleneck_triggered.evidence_sweep.spiral_fixed_causal audit \
  --output RESULTS/controlled_bt/spiral_fixed_causal_v1

python -m experiments.bottleneck_triggered.evidence_sweep.spiral_fixed_causal analyze \
  --output RESULTS/controlled_bt/spiral_fixed_causal_v1
```

If execution stops partway through, use the **same exact flags**, add `--resume`, and do not alter the source commit, device, calibration cap or model-saving setting. The runner refuses mismatched study plans and corrupted/changed per-run manifests. Never mix the existing 255-run endogenous grid with the fixed-event study output.

## Audit requirements

Check **45/45** completed manifests, no failed runs, exact event epochs [15, 30] for every intervention arm and none for identity, 50 epoch logs, exactly equal training steps/examples and split metadata for each paired seed, test examples used for selection = 0, validation trajectories, and output equivalence with tolerance `1e-4` in coupled mode. The positional mode is expected to change full-width predictions at an intervention, which is recorded rather than rejected.

Capture the immediate `full_width_output_max_diff` and `no_gate_output_max_diff` for both interventions. Validation at the intervention epoch is measured *before* the intervention; the following epoch is not an instantaneous post-intervention causal measurement. Do not mislabel it.

The analyzer writes into `causal_analysis/`: `paired_seed_deltas.csv`, `paired_contrast_summary.csv`, `validation_trajectories.csv`, `intervention_events.csv`, `analysis.json`. The primary contrast uses exact paired sign-flip enumeration (2^5 permutations), for which minimum two-sided p = 0.0625. Bootstrap intervals are descriptive; the five paired seeds and multiple secondary comparisons prevent strong population-level claims.

## Stop conditions

Stop and report rather than silently repair: source SHA mismatch, dirty/unsafe checkout, corrupted or changed dataset splits, tests failing, missing/duplicate configurations, missing event, unequal budgets, mismatched manifests, missing or nonfinite training metrics, coupled-gate equivalence failure, test leakage. Preserve all previous outputs; do not merge to `main` or push new local changes without permission.
