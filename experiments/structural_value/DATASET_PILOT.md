# Dataset-backed AWN pilot — engineering verification

Reference datasets: Spiral and SpiralHard, with original repository
DATA_SPLITS fold files. The original dataset.py generator and original AWN
model/ELBO are used; this is not a synthetic stand-in for either benchmark.

## Predeclared CI pilot

- Both reference tasks, initial seed 42
- Checkpoint epoch 3 (observed, not tuned against final outcomes)
- 8 post-checkpoint training optimizer steps per treatment (short smoke)
- 5 actions: skip, seeded random positional, W positional, G positional,
  and W neuron-coupled control
- Intermediate horizons H=1,4,8 steps
- Development calibration and validation strictly disjoint from training,
  and test labels excluded from action selection and teacher labels
- Risk: held-out development cross-entropy
- Online cost: measured wall seconds. Cost weight=0.01 per reference second.
  This is a declared engineering exchange rate, not a statistically tuned cost.
- Raw observations, teacher labels, manifest and independent audit report
  are uploaded by GitHub Actions as per-dataset artifacts.

Run locally from the repo root with the reference AWN dependencies:

    python -m experiments.structural_value.scripts.pilot \
      --dataset Spiral --seed 42 --checkpoint-epoch 3 \
      --horizon-epochs 1 --max-rollout-steps 8 \
      --rollout-seed 12345 --coupled-control \
      --cost-weight 0.01 --reference-seconds 1.0 \
      --output outputs/structural_value/spiral_engineering

    python -m experiments.structural_value.scripts.audit_pilot \
      --input outputs/structural_value/spiral_engineering \
      --output outputs/structural_value/spiral_engineering/audit.json

The audit rejects test leakage, mismatched model/split/seed metadata,
missing/duplicated treatment endpoints, unequal optimizer steps and
training examples, malformed costs, and incorrect label signs or value
formulas. No ranking is treated as confirmatory.

**Do not interpret eight steps at one checkpoint/seed as evidence of a
generalizable intervention policy.** This is a dataset integration
smoke test, followed by a separately declared larger multi-checkpoint
study and eventual leakage-safe held-out test.
