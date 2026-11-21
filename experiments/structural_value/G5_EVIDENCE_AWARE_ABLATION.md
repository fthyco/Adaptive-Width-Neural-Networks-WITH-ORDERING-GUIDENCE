# G5A — Evidence-Aware Action-Value Ablation (Fresh Same-Checkpoint Design)

Status: implementation + new-seed exploratory evaluation. G4 frozen confirmation is unchanged.

## Why we rejected historical G2 replay

An initial G2 feature-only replay was proposed to recover feature vectors
after the original labels existed. Although all Spiral cohorts matched, the
SpiralHard baselines differed numerically from the original G2 trajectories
(e.g., rate 0.0256877 vs 0.0256082 at epoch 20). Those states are
**not identical counterfactual contexts**. We explicitly rejected
pairing such features with G2 labels. There is no tolerance relaxation
or selective seed exclusion. Failed G5A historical-replay CI:
[run 44856367948](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44856367948).

## New G5A experimental design

- **Datasets:** Spiral, SpiralHard.
- **Fresh seeds:** 20331, 20332, 20333. None are G2 or G4 seeds.
- **Checkpoints:** epochs 5,10,20,35 on uninterrupted original AWN.
- **Actions:** SKIP, random positional, weight positional, gradient positional.
- **Matched future budget:** 8 optimizer steps, intermediate H=1,4,8.
- **Paired labels:** CE_H(skip) - CE_H(action) - 0.01 * incremental action online seconds.
- **96 treatment continuations**, **288 audited observations** over 24 checkpoints.
- **Before any intervention:** measure W and G evidence on the frozen
  calibration fold of a copied native-AWN checkpoint; compute the exact
  deterministic random ordering; save eight features for the proposed
  permutations. These feature records are written before any future
  training examples or counterfactual outcomes are generated.
- **Effective prefix semantics:** AWN's quantile-derived learned width may exceed physical\n  hidden size at a checkpoint. A permutation can only move existing neurons,\n  so prefix-replacement features use `min(learned_width, physical_width)`;\n  the original learned-width feature remains unchanged. This is a fixed\n  measurement-contract correction, not a width-learning modification.\n- **Same checkpoint identity:** future action forks restore the *same*
  in-memory captured model + optimizer + RNG snapshot from which
  the evidence was computed. No cross-run model-state matching.
- **Model comparison:** 10-feature G3 Ridge architecture vs 18-feature
  evidence-aware Ridge architecture. Both **retrained only on training
  groups** of the same leave-one-seed-out / leave-one-dataset-out
  splits, with fixed alpha=12 and zero intervention threshold. We
  compare net value, action-wise MAE, and action choices.
- **No final test examples** for teacher labels or feature selection.

## Interpretation limits

This is a fresh-cohort, same-checkpoint **exploratory ablation**, NOT
prospective confirmation of a previously trained evidence-aware policy.
The previous G3 coefficients are not evaluated as the baseline; only
the G3 *architecture* is fitted on the same training fold for fairness.

The original G4 results and locked model are unaffected. G5A still
uses just two related synthetic tasks, three independent seeds, and a
short 8-step horizon. The target includes the action-specific evidence
and intervention time recorded in each fork, but the cost of computing
the additional predecision G/W descriptors **for every decision** has
not been charged to the G5A policy. Therefore a positive value
difference would NOT prove end-to-end deployment superiority.

A final positive claim requires a later prospectively locked model on
untouched seeds/tasks, prepublished predictions, full feature-compute
and controller overhead, and paired independent outcomes.
