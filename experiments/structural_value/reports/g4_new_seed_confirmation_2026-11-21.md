# AWN — G4 Frozen-Policy New-Seed Confirmation

Date: 2026-11-21  
Outcome: **Engineering confirmation complete: 24 committed decisions, 96 native AWN action forks, 288 audited observations. Positive but very small new-seed mean value; not a robust or confirmatory generalization claim.**

## 1. Provenance and temporal integrity

- Original frozen G3 model:
  [successful G3 CI run 44849364139](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44849364139),
  artifact 14683998134.
- Frozen full-history Ridge model SHA-256:
  `26e6950775efce0fb2f6c6461c3c30229a673bb530efa7e417183bb9bedfa46a`.
- Original **before-outcomes** G4 source commit:
  `0aff349d49f1bb1be42161a2b08d291bb309c17b`.
- Original Stage-A prediction artifacts and three successfully confirmed
  Stage-B cohorts: [run 44850446436](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44850446436).
  The overall original workflow **failed**, because three other Stage-B
  jobs rejected bitwise comparison of float64 forecasts from different
  CI CPU environments.
- Strictly targeted numerical-verifier correction: **absolute tolerance
  1e-12 for the predicted float64 values, while the selected action
  remains required to match exactly**. Original frozen predictor
  parameters, six prediction artifacts, selected actions, snapshot
  hashes, datasets and seeds were **not changed**. Added explicit tests
  confirming failure on larger numerical changes, changed action or
  missing target. Preflight **22/22 passed**.
- [Successful recovery + full aggregate run 44851313930](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44851313930).
  It reused the three original *successful* Stage-B results, and ran
  **only the three previously failed Stage-B cohorts** using their
  original published Stage-A predictions and model/optimizer/RNG
  snapshots. No prediction was retrained, no new seed was selected,
  and no previously successful outcome was rerun for selection.
- Machine-readable final aggregate:
  [artifact of run 44851313930](https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44851313930/artifacts/14684689436).

The numerical-verification code revision and the original prediction
commit are recorded **separately**. Intervention and outcome code
retained its algorithm; only the cross-environment numeric equality
check and provenance handling were corrected. This is an engineering
repair, **not** a tuning or a new research hypothesis.

## 2. Locked experimental protocol

| Dimension | Predeclared and implemented |
|---|---|
| Reference datasets | Spiral, SpiralHard (original AWN generator and fold files) |
| Previously unseen training seeds | 20271, 20272, 20273 |
| Native AWN baseline checkpoints | Epochs 5, 10, 20, 35 |
| Locked model | G3 full-data Ridge; alpha=12, train-only G3 z-score, no G4 fit |
| Predictor inputs | Nine predecision AWN state features + known dataset identity |
| Decision rule | Highest positive predicted incremental net value; otherwise SKIP |
| Actions actually measured per checkpoint | Original AWN SKIP, Random positional, W positional, G positional |
| Future optimizer-step horizon | 8 (with intermediate H=1,4,8 observations) |
| Engineering cost trade-off | 0.01 CE units per additional recorded online second |
| Future minibatches | Pre-materialized same train-only stream per checkpoint and treatment |
| Final test set used | 0 examples |

**Counting**: 2 tasks × 3 seeds × 4 checkpoints = 24 new decisions.
Each decision has four independently forked continuations: **96
final-H8 rollouts** (24 skip, 72 interventions). Three measured horizons
produce **288 audited paired observations and 288 teacher labels**.
Every fresh seed's 4 checkpoints share one uninterrupted baseline AWN
training trajectory and must not be considered independent replicates.

The G3 model/decision rule was locked from historical data **before**
running any G4 future action branches. All 24 G4 predictions were
published as immutable job artifacts before Stage B began. The original
test fold never contributed labels to training, calibration or
validation used to choose actions.

## 3. Actions committed before observing effects

| Dataset | Seed | Epoch 5 | Epoch 10 | Epoch 20 | Epoch 35 |
|---|---:|---|---|---|---|
| Spiral | 20271 | W | SKIP | SKIP | SKIP |
| Spiral | 20272 | W | SKIP | SKIP | SKIP |
| Spiral | 20273 | W | SKIP | SKIP | SKIP |
| SpiralHard | 20271 | W | G | SKIP | SKIP |
| SpiralHard | 20272 | W | G | SKIP | SKIP |
| SpiralHard | 20273 | W | G | G | SKIP |

**Action counts**: SKIP 14, W 6, G 4, Random 0.

## 4. Primary result: fresh-seed net value at H=8

Primary label (as fixed in G2/G3):

`V(s,a) = CE_H(skip) - CE_H(a) - 0.01 [T_H(a) - T_H(skip)]`

All scores are **incremental net values per checkpoint decision**
relative to the native AWN SKIP baseline; larger is better.

| Policy | Mean net value | Mean CE-only gain | Beneficial / 24 | Harmful / 24 | Worst |
|---|---:|---:|---:|---:|---:|
| Always SKIP | 0 | 0 | 0 | 0 | 0 |
| Always Random | -0.00493626 | -0.00492193 | 3 | 21 | -0.01957718 |
| Always W | -0.00229348 | -0.00227782 | 12 | 12 | -0.03080944 |
| Always G | -0.00096402 | -0.00092262 | 14 | 10 | -0.02300323 |
| **Locked G3 Ridge policy** | **+0.00006308** | **+0.00006552** | **5** | **5** | **-0.00252103** |
| Hindsight oracle (non-deployable) | +0.00194086 | +0.00195879 | 16 | 0 | 0 |

The Ridge policy chose SKIP 14 times, so its 10 actual
interventions were **5 beneficial / 5 harmful**. Its positive aggregate
mean is **small**, with a non-negligible downside in individual
contexts, and the oracle is only a hindsight diagnostic.

### By task

| Task | Locked Ridge mean net | Always G | Always W | Always Random |
|---|---:|---:|---:|---:|
| Spiral | +0.00008833 | -0.00296113 | -0.00549518 | -0.00710868 |
| SpiralHard | +0.00003784 | +0.00103308 | +0.00090822 | -0.00276383 |

The locked Ridge policy improves on unconditional intervention rules
**in the pooled 24-decision average**, but crucially **does not beat
Always G or Always W on SpiralHard alone**. This matters for any claim
about generalization beyond the historical G3 tasks.

### By new seed (both tasks together)

| New seed | Ridge mean net value across 8 decisions |
|---|---:|
| 20271 | +0.00002252 |
| 20272 | **-0.00036970** |
| 20273 | +0.00053642 |

One of three held-out training seeds has **negative** learned-policy
mean value relative to SKIP. With just three seeds, this is too little
evidence to claim stability or statistical significance.

### Per-trajectory observed chosen value at each epoch

Entries are observed incremental net values for the **already committed
actions**, not optimistically chosen hindsight treatments.

| Dataset | Seed | H8 epoch 5 | H8 epoch 10 | H8 epoch 20 | H8 epoch 35 |
|---|---:|---:|---:|---:|---:|
| Spiral | 20271 | -0.00074176 (W) | 0 (SKIP) | 0 (SKIP) | 0 (SKIP) |
| Spiral | 20272 | -0.00030450 (W) | 0 (SKIP) | 0 (SKIP) | 0 (SKIP) |
| Spiral | 20273 | +0.00210621 (W) | 0 (SKIP) | 0 (SKIP) | 0 (SKIP) |
| SpiralHard | 20271 | -0.00078781 (W) | +0.00170976 (G) | 0 (SKIP) | 0 (SKIP) |
| SpiralHard | 20272 | -0.00013204 (W) | -0.00252103 (G) | 0 (SKIP) | 0 (SKIP) |
| SpiralHard | 20273 | +0.00011251 (W) | +0.00047216 (G) | +0.00160049 (G) | 0 (SKIP) |

The predicted W intervention at epoch 5 is **not consistently
beneficial**, and the same predeclared G at SpiralHard epoch 10 is
beneficial for two seeds but harmful for seed 20272. These
differences are not attributable to a revised policy, since the
predictions were archived before the outcome stage.

## 5. Comparison with exploratory G3

| Study | Held-out evaluation type | Locked Ridge mean incremental net value |
|---|---|---:|
| G3 | Leave-one-historical-seed-out | +0.00042149 |
| G3 | Leave-one-historical-dataset-out | +0.00065130 |
| **G4** | **Three genuinely new training seeds, predictions precommitted before outcome measurement** | **+0.00006308** |

The measured mean benefit **shrinks markedly** under truly new-seed
confirmation. The two historical G3 protocols should not be pooled
with G4 as if they were new independent samples.

## 6. Research interpretation and limitations

**Established engineering results:**
- Unchanged native AWN learned-width training can be checkpointed,
  and the full optimizer/RNG state restored across independent CI
  phases without mixing the forked training budgets.
- A frozen learned value policy can issue irrevocable **pre-action**
  predictions on new seeds, and its outcomes can be evaluated against
  a matched native AWN SKIP path.
- The positive pooled G4 mean (+0.00006308) is accompanied by
  heterogeneous effects and **one negative seed mean**.
- A fixed policy that frequently abstains can outperform
  unconditional intervention in the pooled empirical average under
  the declared H8 engineering metric, but the margin is very small.

**Not established:**
- Statistically reliable or broad out-of-sample generalization.
- Benefit on unseen task families or data distributions.
- Reliable online computational savings; the 0.01-per-second cost
  exchange rate is a modeling convention and CPU timing noise was
  not independently repeated.
- Benefits from resizing width or jointly learning width and
  intervention action. G4 tests positional reordering only.
- That the checkpoint epochs are bottlenecks or optimal trigger times.
- Confirmatory final-test performance; the final test fold remains
  untouched for any policy selection.

**Primary conclusion:** G4 confirms that the conditional intervention
problem is measurable prospectively without using future labels for
action selection. **It does not yet establish a robust net improvement
over no intervention**. The central research question remains open
for broader tasks, horizons, seeds and independently replicated cost
measurements. Do not change G4 seeds, policy threshold or hyperparameter
to produce a larger result after observing this confirmation.

## 7. Next science gate

The next experiments require a new, prospectively declared setup:
longer future horizons, additional dataset families, new seeds,
a stable compute-cost measurement protocol and independent uncertainty
quantification by training trajectory. In parallel, consider measuring
CE-only gain separately from noisy online-wall-time cost. Do not mix
these future new study data into G4 or revise this confirmation's
model based on its observed teacher labels and still call it untouched.
