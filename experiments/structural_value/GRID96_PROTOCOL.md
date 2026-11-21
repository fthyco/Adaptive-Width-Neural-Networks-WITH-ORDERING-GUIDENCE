# AWN Structural Value — 96 Counterfactual Fork Engineering Grid

## Predeclared design

This extends the completed single-checkpoint Spiral and SpiralHard smoke pilots.
Original AWN / DynamicMLP / ELBO and original dataset.py generators and
DATA_SPLITS folds stay unchanged. Results are **not** a trained predictor.

| Dimension | Fixed choice |
|---|---|
| Datasets | Spiral, SpiralHard |
| Model / split seeds | 42, 123, 456 |
| Baseline train checkpoints (epochs) | 5, 10, 20, 35 |
| Actions | skip, random positional, W positional, G positional |
| Future rollout streams | 1 deterministic stream per checkpoint |
| Future optimizer-step horizon | 8 |
| Recorded intermediate endpoints | H = 1, 4, 8 |
| Cost exchange rate | 0.01 CE per reference second |
| Gate semantics | positional for all four actions |
| Test set for labels | 0 examples |

**Counting**: 2 datasets × 3 seeds × 4 checkpoints × 4 actions × 1
stream = **96 independent model/optimizer-fork continuations** (including
24 skip baselines). At three evaluation horizons, 288 raw observations
and paired labels are expected. There are 72 non-skip treatments, not 96.

Epoch positions are **temporal strata**, not verified plateau labels. Do
not post-hoc relabel epoch 20 or 35 as a detected bottleneck, and do not
select a checkpoint after inspecting treatment outcomes.

## Efficient execution

For each dataset/seed, train a **single uninterrupted AWN baseline** to
epoch 35, preserving original AWN learned width updates. At each of the
four predetermined epochs, capture the entire model/Adam/RNG/sampler state,
then fork skip/random/W/G and continue each branch on **the same immutable
eight-step training batch stream**. Restore the original baseline RNG
(including train-loader sampler) after each set of forks, so counterfactual
evaluation does not alter later checkpoint training.

The experiment records decision-time features before treatment:
training ELBO, development validation CE, width, learned width, rate,
recent development validation improvement, and elapsed optimizer steps.

## Leakage, fairness and provenance

- Development calibration, validation and training are disjoint and
  derived through the existing safe_loaders contract; the original
  test set is not used to train the teacher or build labels.
- All four actions at a checkpoint receive the same steps, examples,
  seed, checkpoint and future batch-stream fingerprint.
- Original AWN width remains adaptive after interventions and may
  evolve differently across treatments.
- W/G evidence costs, permutation overhead, training costs and validation
  costs are recorded separately. Short CPU timing noise means a measured
  advantage of milliseconds is **not** robust evidence of better policy.
- The independent audit validates each per-checkpoint observation/label
  equation, then each four-checkpoint dataset/seed study. Aggregation fails
  unless all six dataset/seed studies pass and share a single code commit.
- The aggregator reports each dataset/action across the twelve
  checkpoint-pairs with mean/median risk difference, improvement counts
  and **three seed-level means plus their descriptive SD**. No
  independence across the four checkpoints of one seed is assumed.
- Reports do not claim significance or transfer on only three seeds.

## Reproduction

Single dataset/seed (on native AWN environment):

    python -m experiments.structural_value.scripts.run_grid \
      --dataset Spiral --seed 42 \
      --output outputs/structural_value/grid_Spiral_seed42

    python -m experiments.structural_value.scripts.audit_grid \
      --input outputs/structural_value/grid_Spiral_seed42 \
      --output outputs/structural_value/grid_Spiral_seed42/grid_audit.json

The dedicated GitHub workflow runs all six dataset/seed combinations,
uploads each raw training history / 4 checkpoint subfolders / per-run
audit, and uploads a complete aggregate only if all six jobs pass.

The completed first smoke study is recorded separately in
reports/dataset_pilot_smoke_2026-11-21.md. The current study increases
coverage but retains an **eight-step engineering horizon**.

## Scientific next gate

Audit signal variation across seeds and checkpoints, no-op rates and
cost sensitivity before constructing a predictor. A predictor must
be evaluated using group-held-out dataset/seed/checkpoint splits,
compared with skip/random/W/G baselines and calibrated prospective
decision value. Untouched final test is reserved for confirmation
after method selection. No predictor, learned trigger, resize/joint
action or RL controller is part of this engineering grid.
