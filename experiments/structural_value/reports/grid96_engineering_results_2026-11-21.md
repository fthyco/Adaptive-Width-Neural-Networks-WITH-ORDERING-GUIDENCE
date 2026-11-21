# AWN Structural Adaptation — 96-Fork Engineering Grid Results

Date: 2026-11-21  
Research status: **COMPLETED engineering grid; no learned intervention-value predictor or confirmatory evidence**.

## 1. Verified source and evidence

- Source commit: \`ffb347214a86e4fbf92912ff2bf4dac54039edd5\`
- Successful GitHub Actions run:
  https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44848340275
- Complete machine-readable aggregate artifact:
  https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44848340275/artifacts/14684442179
- Six raw per-dataset/seed artifacts are attached to the same CI run.
- Independent design-contract tests: **9 passed**.
- All six real-AWN dataset/seed grid jobs passed, as did the final
  fail-closed aggregate job. No unverified or missing groups were accepted.
- Native original AWN/DynamicMLP + ELBO, original Spiral/SpiralHard
  dataset generation and repository fold splits. Original AWN modules
  and \`main\` remain unchanged.

## 2. Predeclared experimental grid

| Dimension | Value |
|---|---|
| Datasets | Spiral, SpiralHard |
| Training/split seeds | 42, 123, 456 |
| Fixed checkpoint epochs | 5, 10, 20, 35 |
| Actions per checkpoint | Skip, Random positional, Weight positional (W), Gradient positional (G) |
| Post-checkpoint future streams | 1 matched stream |
| Future horizon | 8 optimizer steps |
| Observed horizons | 1, 4, 8 steps |
| Decision-time features | Epoch, width, learned width, exponential rate, train ELBO, validation CE/accuracy, recent CE trend, prior optimizer steps |
| Risk metric | Held-out development validation cross-entropy |
| Cost | Synchronized online wall seconds; declared weight = 0.01 CE per reference second |
| Final test used in treatment decisions/labels | 0 examples |

**Accounting:** 2 datasets × 3 seeds × 4 checkpoints = 24 checkpoint
decisions. Four independent checkpoint continuations per decision give
**96 final-horizon rollouts** (24 untreated Skip, 72 treatments). Three
recorded horizons give **288 matched observations and 288 teacher labels**.

For each dataset/seed, the model was trained once along a continuous
baseline, not retrained from scratch separately for each checkpoint.
Checkpoint forks restored model, optimizer, global RNG and sampler; after
the forks, the live baseline's RNG and loader generator were restored.
All actions within a checkpoint received identical training batches,
optimizer-step count and example count. Each checkpoint was separately
audited for test leakage, pairing, label formula, width and resource budget.
All six jobs shared the same source commit.

Checkpoint labels are **fixed temporal positions**, not verified plateau
or bottleneck classifications.

## 3. Outcomes at horizon H = 8

Define \`Δ Risk = CE(Skip) − CE(Action)\`, so positive values indicate an
improvement by the intervention. Averages and medians are descriptive
across four checkpoints × three seeds within each dataset.

### Spiral

| Action | Mean Δ Risk | Median Δ Risk | Mean Net Value | Improved / 12 | Seed-mean SD |
|---|---:|---:|---:|---:|---:|
| Skip | 0 | 0 | 0 | — | 0 |
| Random | -0.01295464 | -0.01095538 | -0.01298141 | 0/12 | 0.00118843 |
| W | -0.00555238 | +0.00138766 | -0.00558202 | 7/12 | 0.00571868 |
| G | -0.00506926 | +0.00056581 | -0.00511757 | 7/12 | 0.00514334 |

### SpiralHard

| Action | Mean Δ Risk | Median Δ Risk | Mean Net Value | Improved / 12 | Seed-mean SD |
|---|---:|---:|---:|---:|---:|
| Skip | 0 | 0 | 0 | — | 0 |
| Random | -0.00605222 | -0.00656128 | -0.00607894 | 1/12 | 0.00212157 |
| W | -0.00078767 | -0.00000852 | -0.00081621 | 6/12 | 0.00115268 |
| G | -0.00012183 | +0.00042139 | -0.00018910 | 7/12 | 0.00099744 |

### Descriptive per-seed averages (Δ Risk)

| Dataset | Action | Seed 42 | Seed 123 | Seed 456 |
|---|---|---:|---:|---:|
| Spiral | Random | -0.01347346 | -0.01159501 | -0.01379545 |
| Spiral | W | -0.00631331 | +0.00050867 | -0.01085250 |
| Spiral | G | -0.00582666 | +0.00041079 | -0.00979190 |
| SpiralHard | Random | -0.00369228 | -0.00780147 | -0.00666292 |
| SpiralHard | W | -0.00146043 | +0.00054330 | -0.00144587 |
| SpiralHard | G | -0.00124802 | +0.00065020 | +0.00023235 |

No treatment's mean risk improvement or mean net value is positive on
either dataset. **Skip wins on average in this engineering grid.** Among
the tested interventions G has the least negative mean on both datasets;
Random is the worst in mean risk on both.

However, W and G have locally positive checkpoint-level results despite
negative overall means. Especially on Spiral, the median of their risk
differences is positive while the mean is negative. This indicates
**heterogeneous action outcomes**, including harm that can outweigh
smaller local improvements. The variance between seed-level averages
reinforces that a simple unconditional intervention rule is not justified.

## 4. What the results do and do not establish

**Established:**
1. Original AWN can be run through genuine dataset-backed checkpoint
   forks and the original ELBO while automatically learning width.
2. Counterfactual treatment-vs-skip teacher labels can be paired, counted
   and validated with equal training budgets.
3. Effects differ across measured checkpoint/seed/task contexts.
4. Neither W nor G gives an unconditional average improvement over Skip
   under the declared 8-step setup.

**Not established:**
- That one can predict positive future action value from available
  decision-time features; **no predictor has been trained or evaluated**.
- That the selected chronological checkpoints identify true bottlenecks.
- That the performance differences are statistically significant or
  generalize beyond the two tasks, three seeds and short horizon.
- That wall-time cost differences of milliseconds are stable across
  hardware and repetition.
- That a policy choosing the retrospectively best treatment on these
  same validation labels would work out of sample. Such an oracle would
  be optimistic and cannot count as generalization evidence.
- That learned resize/joint decisions, a controller or RL improve AWN.

The four checkpoints in one seed share one baseline training trajectory
and **are not independent replicates**. The reported between-seed SD is
computed from just three seed-level means per dataset/action; it is not
a confidence interval or significance test.

## 5. Next research gate

Before any prospective decision-value claim:

1. Examine risk and net-value labels *by checkpoint and seed* and
   characterize harmful intervention tails and no-op prevalence.
2. Predeclare cost-sensitivity and future-horizon analyses; repeat wall
   timing before claiming net cost benefits.
3. Train a deliberately simple value predictor using only decision-time
   information, with grouped holdout to prevent checkpoint or seed leakage.
4. Compare predicted-policy outcomes against Skip, Random, W, G and
   majority/always-intervene rules on groups not used to train or tune.
5. Reserve untouched final-test confirmation until all policies,
   thresholds and data splits are locked.

The research target is **prospective value of structural adaptation**,
not merely proving that an importance ranking can change the network.
