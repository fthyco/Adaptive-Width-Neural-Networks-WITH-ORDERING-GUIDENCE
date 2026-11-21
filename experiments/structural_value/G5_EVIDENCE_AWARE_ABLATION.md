# G5A — Predecision Evidence / Proposed-Change Feature Ablation

Status: exploratory research extension, distinct from immutable G4 prospective confirmation.

## Question

Does giving the action-value regressor direct access to the *proposed
structural change* improve held-out prediction and action choice compared
with original G3 features (epoch, width, training and validation history)?

## Frozen measurement design

- Historical teacher targets: exact, previously audited G2 H8 paired labels
  (source commit `ffb347214a86e4fbf92912ff2bf4dac54039edd5`, run
  `44848340275`). Do not create new labels or use G4 outcome labels.
- Feature replay: independently train original, unmodified AWN baselines
  on G2's Spiral/SpiralHard and seeds 42/123/456. Verify each replayed
  checkpoint state against original G2 to 1e-5 relative, 1e-6 absolute;
  exact identity, width and step matching are mandatory. Abort on mismatch.
- Proposed actions (exactly G2): deterministic random positional order,
  weight-magnitude positional order, gradient-sensitivity positional order.
- New pre-action features: normalized rank displacement, fraction of the
  AWN learned prefix replaced for each proposed action; top-quartile share
  of nonnegative W/G evidence. No future loss or outcome is used to compute them.
- The learned width at this checkpoint defines the prefix. When the
  learned width equals physical width, prefix replacement is zero.
- Both baseline G3 and augmented G5A use **fixed alpha=12 multioutput Ridge**
  with identical targets, unpenalized intercept and train-only standardization.
  The alpha is not selected based on G4 or G5 outcomes.
- Same leave-one-seed-out and leave-one-dataset-out fold membership.
  In all folds whole trajectories, not checkpoints, are held out.
- Compare held-out paired policy net value vs SKIP and baseline G3,
  decision counts and action-wise MAE. Report both positive and negative
  values. No post-hoc feature selection or threshold tuning.
- No final test fold data, no optimizer steps beyond original baseline,
  no mutated AWN or G4 artifacts.

## Interpretation and limitations

This is **retrospective, exploratory G2 feature ablation**, not a new-seed
confirmation. The G2 label set was already observed when designing the
feature test, so results cannot prove unbiased generalization or novelty.
The effective historical units are only six training trajectories, and
adding eight features can overfit. Any positive G5A result requires a
new prospective study with frozen trained coefficients, untouched seeds,
prepublished decisions and paired outcome measurements, with the cost
of computing gradient evidence included in online cost.

Feature collection in G5A replays the frozen historical checkpoints to
recover information not preserved in the original G2 report. It must
never silently pair different network states. A mismatch invalidates
this replay method and requires redesign rather than tolerance hunting.
