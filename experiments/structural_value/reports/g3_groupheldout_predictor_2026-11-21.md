# AWN Structural Value — G3 Group-Held-Out Predictor

Date: 2026-11-21  
Status: **Engineering/exploratory predictor evaluation complete. Not a confirmatory result.**

## Provenance

- Evaluator source commit: `feae9f48500f3656e0a25ef4f2c8e4b592d5ed55`
- Successful original-data CI run: https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44849364139
- Full recorded predictions, labels, fold models and decisions:
  https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44849364139/artifacts/14683998134
- Source G2 run: https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44848340275
- Frozen G2 source commit: `ffb347214a86e4fbf92912ff2bf4dac54039edd5`
- **15/15 predictor contract tests passed.** G2 source raw records were
  downloaded and independently re-audited for checkpoint, budget, seed,
  source commit, validation-vs-test separation and paired label equations.
  No test labels were included in the features.

## Objective

Estimate the expected *incremental future value of action a* at state s
using only state information available before an intervention:

`V(s,a) = CE_H(skip) − CE_H(a) − 0.01 [T_H(a) − T_H(skip)]`

CE_H: predictive cross-entropy after the matched H=8 optimizer-step
continuation. T_H: measured online seconds under the recorded AWN
engineering protocol. The fixed 0.01 cost weight was carried from G2.
**Milliseconds of wall-time difference on a single CPU run have
substantial noise and should not be mistaken for calibrated compute savings.**

Actions: SKIP, random positional, W positional, G positional.

## Model and anti-leakage protocol

- One 3-output multi-target Ridge, predicting action values for Random,
  W and G, and comparing against SKIP value fixed to zero.
- Fixed alpha=12, unpenalized intercept and train-only z-score features;
  no outer-fold hyperparameter tuning.
- Nine numerical decision-time AWN features: epoch, physical width,
  learned width, exponential width rate, train ELBO, development validation
  CE, development validation accuracy, recent validation CE change, and
  baseline optimizer steps elapsed. Plus a known dataset binary indicator.
- No seed ID as predictor input and no action outcome, future width,
  hindsight oracle or label used as a decision-time feature.
- Zero abstention threshold: SKIP if every predicted intervention value
  is nonpositive; otherwise take the highest predicted action value.
- Entire dataset×seed baseline trajectories held together in each split:
  checkpoint epochs 5,10,20,35 from the same model **never** appear on
  opposite sides of the fit/evaluation boundary.

Two separate evaluation protocols were run:

1. Leave-one-seed-out across **both** datasets. 3 folds, 16 training
   checkpoints and 8 held-out checkpoints per fold. Each of the 24
   decisions was evaluated once out-of-seed.
2. Leave-one-dataset-out. 2 folds, 12 training checkpoints and 12 held-out
   checkpoints per fold. Each decision was evaluated once out-of-dataset.

The same 24 checkpoints appear under both protocols, so **do not**
treat the 48 fold-test records as independent evidence.

## Results

All entries below are the *mean observed H=8 incremental net value per
held-out checkpoint decision* relative to original AWN SKIP.

| Policy | Leave-one-seed-out | Leave-one-dataset-out |
|---|---:|---:|
| Always SKIP | 0 | 0 |
| Always Random | -0.0095301722 | -0.0095301722 |
| Always W | -0.0031991176 | -0.0031991176 |
| Always G | -0.0026533324 | -0.0026533324 |
| Train-only mean-best fixed action | 0 | 0 |
| **Fixed Ridge value policy** | **+0.0004214876** | **+0.0006512969** |
| Hindsight oracle (not deployable) | +0.0012570641 | +0.0012570641 |

Constant baselines have the same means under the two protocols because
each protocol holds out every one of the same 24 decisions exactly once.
This does **not** mean the ridge models or their training sets are equal.

### Leave-one-seed-out (seed-group holdout)

- SKIP selected in **15/24** decisions.
- W selected 5 times; G selected 4; Random never selected.
- Of the 9 selected interventions, **7 positive, 2 negative**.
- Worst single selected net value: **-0.0008504705**.
- Seed-heldout mean net values:
  - seed 42: +0.0001832989
  - seed 123: +0.0002587926
  - seed 456: +0.0008223714
- Held-out value-prediction MAE:
  - Random: 0.0032217030
  - W: 0.0053891204
  - G: 0.0052177696

### Leave-one-dataset-out (dataset-group holdout)

- SKIP selected in **13/24** decisions.
- W selected 6; G selected 5; Random never selected.
- Of 11 selected interventions, **10 positive, 1 negative**.
- Worst single selected net value: **-0.0003950499**.
- Dataset-heldout mean net values:
  - Spiral: +0.0008412421
  - SpiralHard: +0.0004613518
- Held-out value-prediction MAE:
  - Random: 0.0050824611
  - W: 0.0058202616
  - G: 0.0055104881

The best-action hindsight oracle has no observed harmful decisions by
construction and **uses labels from the held-out decision**; therefore
it is an upper bound/diagnostic, not a feasible research baseline.
The learned Ridge policy did **not** attain the oracle.

## Interpretation

The experiment provides a **positive exploratory result** for
learning a context-conditional decision about positional reordering
instead of unconditionally reordering every checkpoint. The held-out
Ridge policy has positive mean net value relative to zero SKIP under
both predeclared *offline* protocols on the 24 historical G2 states,
while the three always-intervene policies have negative means.

This does **not** establish external predictive validity:
- Only 2 tasks and 3 seeds exist; four checkpoints within each seed
  are correlated and not independent datasets.
- The cost label is hardware- and run-dependent; net-value margins
  are small, and no repeated timing variance was measured.
- Each held-out outcome still comes from the previously recorded
  counterfactual G2 dataset, not genuinely new future trajectories
  after model/policy selection.
- The same historical data motivated the method and were used for
  exploratory engineering and fixed-model design. These outcomes
  must not be counted as an independent untouched confirmation.
- No statistics, standard error or confidence bounds for prospective
  policy generalization are established.
- No learned width modification, joint width/order action, or RL
  controller is implemented by this experiment.

## Next research gate (confirmation)

Freeze the model class, features, alpha, zero threshold and action set.
Then acquire **new independently seeded AWN trajectories** not used
in G2/G3 model development. Predict from checkpoint states *before*
running any treatment, and only afterward collect matched
counterfactual labels. Evaluate against native SKIP, always-W/G and
train-only constant baselines without outcome-based tuning. Repeat
cost timings or report CE-only gain as the primary robust signal.
Keep the final held-out test untouched until the policy is frozen.

Only if the locked policy succeeds on genuinely new trajectories
should the study progress from exploratory evidence to a prospective
generalization claim.
