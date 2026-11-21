# Gate G3 — Prospective Intervention Value Predictor (first baseline)

## Prior work and source data

Use the **immutable, completed G2 AWN GRID96** source:
- Code commit: `ffb347214a86e4fbf92912ff2bf4dac54039edd5`
- Actions run: https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44848340275
- 2 reference tasks, seeds [42,123,456], checkpoint epochs [5,10,20,35],
  actions skip/random/W/G, native AWN ELBO, H=8 optimizer steps.
- Complete paired observation/teacher label files and independent auditors
  are re-run on input. All six dataset×seed trajectories must be present.
- The untouched final test set is **never** accessed by this evaluation.

## Mathematical target and predictor

At a checkpoint with state features `x`, predict the (future) net
value of each intervention `a` against the same-checkpoint AWN skip:

`V(x,a) = CE_H(skip) - CE_H(a) - 0.01 * (T_H(a) - T_H(skip))`

The factor `0.01` is the previously declared *engineering* CE-per-second cost
exchange rate, not an optimized or robust hardware cost model. Every label is
from an independently forked, matched 8-step AWN continuation.

Frozen predecision features:
- checkpoint epoch, physical width, learned width, width rate
- train ELBO, development validation CE, validation accuracy
- validation CE change over the past five epochs, steps elapsed
- dataset identity (SpiralHard vs Spiral)

No future reward, selected outcome, label, validation on the held-out
group, action-induced width or hindsight best action enters the features.
Seeds are *group identifiers*, not predictor inputs.

First predictor is **multioutput Ridge**, with training-only z-score
standardization, unpenalized intercept and fixed L2 alpha = 12.0.
No inner hyperparameter search, post-hoc threshold tuning or deployment
claim. The action with the highest positive predicted net value is
selected; otherwise **SKIP**. Ties select SKIP.

## Two independent descriptive holdout protocols

1. **Leave-one-seed-out**: hold out an entire seed for *both* datasets
   (8 decisions held out, 16 trained, 3 folds).
2. **Leave-one-dataset-out**: hold out all 3 seed trajectories of one task
   (12 held out, 12 trained, 2 folds).

These are two *evaluations of the same 24 decision units*, not 48
independent observations. All checkpoints from one dataset×seed
trajectory stay within one split. Standardization and ridge fitting
happen only within that fold's training trajectories.

Full-information retrospective outcomes allow exact held-out action
evaluation on the *recorded* H8 forks without off-policy importance
weighting; this is **offline evaluation**, not a new prospective run.

## Predeclared baselines and reporting

- always SKIP (zero incremental value)
- always Random, always W, always G
- constant train-mean-best intervention with zero abstention threshold
- fixed ridge value predictor + zero abstention threshold
- hindsight oracle (nondeployable upper bound, shown **only** as diagnostic)

Report: mean/median value, harmful and beneficial decision counts, worst
harm, predicted action frequencies, per-action held-out absolute value
prediction error, plus individual fold outcome/coefficients. Do not
pool leave-seed and leave-dataset outcomes for significance testing.
Report both positive and negative results.

## CI and executable command

The predictor workflow fetches the original 96-fork GitHub Actions
artifacts by **run ID** from G2, recomputes all raw-data audits and
verifies the source commit before fitting anything.

    python -m experiments.structural_value.scripts.value_predictor \
      --root outputs/structural_value \
      --output outputs/structural_value/predictor_groupheldout.json

This stage cannot establish reliable wall-time cost superiority, trained
policy generalization beyond two small datasets, causal benefits under
unseen real deployments, or substantive statistical significance.
A final candidate refit on all 24 decisions is exported but expressly
marked **not independently validated**.

The next scientific test must use fresh trajectories or predeclared
independent confirmation after policy/thresholds are locked.
