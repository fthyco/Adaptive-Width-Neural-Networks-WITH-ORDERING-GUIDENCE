# AWN — G5A Evidence-Aware Action-Value Ablation

Date: 2026-11-21
Status: **Engineering complete; exploratory evidence-aware feature improvement NOT supported.**

## 1. Question

Do pre-action W/G/random positional proposal descriptors improve prospective
H8 structural intervention-value prediction over the original G3 feature
family (epoch, native AWN width, training and validation history)?

This is an **exploratory grouped-holdout ablation**, not a newly frozen
prospective policy and not an independent confirmation of structural-value
generalization.

## 2. Provenance and audit trail

- Original historical G2 replay attempt:
  [run 44856367948](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44856367948).
  Replay failed strict same-checkpoint validation on SpiralHard, including
  target width-rate 0.025608226657 vs replay 0.025687700137 at epoch 20,
  seed 42. Historic G2 features and labels were **not** paired.
  We did not alter matching tolerance to force an exploratory positive result.
- New G5A study: native AWN training on new seeds 20331,20332,20333, with
  preaction proposal features computed from **the exact in-memory snapshot**
  subsequently used for four action forks at that checkpoint. The original
  learned-width feature is not clamped. Only the physically available prefix
  used by the proposal descriptor is capped at physical width, as required
  when the AWN quantile exceeds currently allocated neurons.
- Original G5A new-cohort run:
  [44857181035](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44857181035),
  original same-snapshot source commit
  `321263d3d7785cc72ad8afce0f24ab23e77b367e`. All six
  cohort jobs completed successfully, each with four preaction checkpoints,
  16 final-H8 forks and 48 audited horizon observations. The overall
  original run **failed only at JSON report serialization**:
  a G2-era summarizer selected old seeds 42/123/456 rather than the
  observed G5 seed IDs, generating empty-group NaNs.
- Narrow recovery: derive actual trajectory IDs from the audited rows,
  enforce six complete trajectories, add a strict JSON finite-value test,
  reuse **all six original successful cohort artifacts unchanged**.
  No training, intervention or teacher outcome was rerun and no model
  hyperparameter was changed after observing labels.
- [Successful aggregate recovery 44857553659](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44857553659),
  aggregation source commit `624c06c08df333ed1af4c72e34674b1af49d3862`.
- [Machine-readable complete recovered aggregate](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44857553659/artifacts/14688366865).

## 3. Experimental design

| Dimension | Fixed G5A design |
|---|---|
| Original reference network | Unmodified AWN one-hidden-layer MLP |
| Datasets | Spiral, SpiralHard |
| New seeds | 20331, 20332, 20333 |
| Native AWN checkpoints | Epochs 5, 10, 20, 35 |
| Decision contexts | 24 across 6 uninterrupted training trajectories |
| Matched actions | SKIP, Random positional, W positional, G positional |
| Counterfactual continuation | 8 optimizer steps, H1/H4/H8 observations |
| Final-H8 branch outcomes | 96 |
| Audited horizon observations | 288 |
| Label | CE(skip) - CE(action) - 0.01 × incremental branch online seconds |
| Baseline model | Train-only standardized 10-feature multioutput Ridge, alpha 12 |
| Augmented model | Same Ridge alpha 12 + 8 action-proposal evidence features |
| Evaluation | Leave-one-seed-out and leave-one-dataset-out grouped holdouts |
| Final test examples used for labels/selection | 0 |

G5A does **not** compare frozen historic G3 model coefficients against a new
model. It compares the original G3 **feature architecture re-fitted on
each G5A training fold** with the evidence-augmented architecture, so
both use identical new-cohort train/heldout divisions and targets.

Eight measured features: for Random, W and G separately, mean normalized
rank displacement and fraction of the AWN currently available active
prefix replaced; for W and G additionally, top-quartile concentration
of nonnegative neuron evidence scores. W/G scores come from the
existing evidence methods on calibration-only samples before interventions.

## 4. Primary empirical comparison

Incremental net value is realized H8 CE gain minus declared branch online
seconds cost, relative to SKIP (=0). Larger is better. **This label excludes
the extra decision-time feature extraction incurred by the augmented
predictor; therefore this is NOT fully accounted deployment net value.**

| Group-heldout evaluation | G3-style 10 inputs | G5A 18 inputs | G5A minus G3 |
|---|---:|---:|---:|
| Leave one fresh seed out | +0.0002865126 | +0.0002883966 | **+0.0000018840** |
| Leave one task/dataset out | +0.0000451667 | **-0.0009439860** | **-0.0009891526** |

The seed-heldout difference is effectively negligible; the dataset-heldout
comparison sharply favors the smaller 10-feature model.

### Seed-heldout diagnostics (24 decisions)

| Statistic | G3-style | Evidence-augmented |
|---|---:|---:|
| Beneficial selected decisions | 6 | 6 |
| Harmful selected decisions | 2 | 3 |
| SKIP | 16 | 15 |
| Random | 0 | 0 |
| Weight | 1 | 3 |
| Gradient | 7 | 6 |
| Random value MAE | 0.00370515 | 0.00378863 |
| Weight value MAE | 0.00326750 | 0.00348767 |
| Gradient value MAE | 0.00336740 | 0.00345169 |

All three action-wise MAEs increased slightly with the eight extra
features in the seed-heldout evaluation.

### Dataset-heldout diagnostics (24 decisions)

| Statistic | G3-style | Evidence-augmented |
|---|---:|---:|
| Beneficial selected decisions | 2 | 1 |
| Harmful selected decisions | 1 | 1 |
| SKIP | 21 | 22 |
| Random | 0 | 1 |
| Weight | 0 | 0 |
| Gradient | 3 | 1 |
| Worst selected net value | -0.00064687 | **-0.02365339** |
| Random value MAE | 0.01025694 | 0.01126753 |
| Weight value MAE | 0.00523001 | 0.00500685 |
| Gradient value MAE | 0.00464772 | 0.00474110 |

The evidence-aware model selected one harmful random intervention that
strongly reduced its mean dataset-heldout performance. Its slightly
smaller W-action MAE does not establish better overall predictions.

## 5. What the result does and does not show

**Established:**
- Before-action W/G/random structural proposals can be computed and
  joined to genuinely matched native-AWN forks without using any
  future outcome as a predictor input.
- Fresh-seed raw audits and fair grouped-fold predictor comparisons can
  be executed reproducibly, with exact source/aggregation provenance.
- These eight descriptors do not yield a meaningful robust improvement
  under this small fixed-alpha 24-decision ablation. In particular,
  leave-dataset-out policy value worsened.

**Not established:**
- That structural evidence is intrinsically unhelpful. This tests one
  representation and one linear regularized model, not the broader
  feature hypothesis.
- Predictive transfer to a new task family; Spiral/SpiralHard are related.
- Advantage net of computing W/G features on every real deployment decision.
- Reliability over longer future horizons or across more independent
  trajectories.
- Any improvement to AWN learned width. Only reorder actions were tested.
- That a 24-checkpoint exploratory grouped-heldout study proves causal or
  statistically robust action policy superiority.

## 6. Research decision

**Do not promote the 18-feature G5A model into an adaptive AWN controller
based on these results.** Retain the smaller 10-feature Ridge as the
strong baseline for future controlled studies; report the negative
ablation transparently.

Next research should be *predeclared* on additional independent trajectories
and diverse datasets, with action-conditioned feature representations
rather than indiscriminate concatenation, and evaluation of uncertainty
and all decision-time evidence costs. Do not tune on these 24 G5A
outcomes and then claim a new-seed independent confirmation.

Source design: `experiments/structural_value/G5_EVIDENCE_AWARE_ABLATION.md`.
