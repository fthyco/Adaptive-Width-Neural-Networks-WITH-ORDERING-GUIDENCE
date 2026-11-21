# Structural Value — Counterfactual AWN Research Pilot

Scope: Steps 0–3 of Research Setup v1.0 and Mathematical Formulation v1.0.
This is an isolated experimental extension, **not** a validated predictor.

## Implemented contracts

- Deep model+optimizer fork (single object graph), scheduler, scaler, Python/NumPy/CPU/CUDA RNG, loader generator state, history, provenance.
- Explicit skip/random/W/G actions and positional vs coupled gate modes; resize/joint fail closed.
- Real AWN ELBO train step and unchanged intrinsic width adaptation; real predictive validation CE.
- Reuse existing evidence and permutation/optimizer-state migration with identity-rank controls.
- Materialized matched training-only minibatch streams; endpoint evaluation at each requested optimizer-step horizon.
- Separate accounting for evidence, action, training and validation time; paired labels require matching run/checkpoint/seed/horizon and equal step/example budgets.
- No-op actions are marked and never counted as successful interventions.
- Development validation is distinct from test via the existing safe_loaders function.

## Single checkpoint run (official AWN environment required)

    python -m experiments.structural_value.scripts.pilot \
      --dataset Spiral --seed 42 --checkpoint-epoch 10 \
      --horizon-epochs 5 --rollout-seed 12345 \
      --output outputs/structural_value/spiral_seed42_cp10

Optional: --coupled-control for a separate preservation control.

The script refuses to overwrite an output directory. Records are
observations.jsonl, labels.jsonl, and manifest.json. Outputs must be reviewed
before any 96-rollout debug grid. The pilot does not access test examples for
feature selection or teacher labels.

## Validation

    python -m pytest -q experiments/structural_value/tests/test_checkpoint_and_rollouts.py
    python -m pytest -q experiments/structural_value/tests/test_native_g0.py

The first suite verifies isolated toy contracts; the second verifies original
AWN/DynamicMLP, its ELBO, Adam resize migration and the real paired-runner
skip replays on synthetic train/validation examples. These are engineering
checks, not evidence for learned policies or cross-task transfer.
