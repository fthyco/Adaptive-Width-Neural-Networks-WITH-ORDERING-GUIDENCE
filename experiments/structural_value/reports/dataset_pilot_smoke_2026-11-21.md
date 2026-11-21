# AWN Structural Value — First Reference-Dataset Pilot

Date: 2026-11-21  
Status: **Engineering smoke completed; no confirmatory research claims.**

## Source and reproduction

- GitHub Actions run: https://github.com/fthyco/Adaptive-Width-Neural-Networks-WITH-ORDERING-GUIDENCE/actions/runs/44847432740
- Source code commit producing observations: `7b9e06f4f76f279089f9445ded598f0444cbe67a`
- Base research branch: `exp/evidence-sweep-two-phases`
- Data from the repository's **original Spiral and SpiralHard generators and DATA_SPLITS fold files**.
- Original AWN/DynamicMLP and ELBO, no modifications to original AWN components.
- Development validation / calibration isolated using existing `safe_loaders`.
- Original standard holdout-test examples used in selection/teacher labels: **0**.
- Config: seed=42; checkpoint_epoch=3; 8 post-checkpoint training steps; H={1,4,8};
  rollout_seed=12345; 5 actions; cost_weight=0.01; reference_seconds=1.
- Training budget: exactly 8 optimizer steps and 1,024 examples per action and dataset.
- Wall time depends on CI hardware. Treat differences at millisecond scale as
  **measurement noise** until repeated paired cost trials are completed.
- Separate reference-data audit: both datasets **PASS**; 15 observations
  and 15 paired teacher labels per dataset; 30 observations and 30 labels total.

## Validation cross-entropy and risk difference at H=8

Risk difference is defined as `validation_CE(skip) - validation_CE(action)`;
positive means the treatment has lower CE at this checkpoint.

### Spiral — seed 42

| Action | Gate semantics | Validation CE | Delta CE vs skip | Net value (declared cost scale) | Final width |
|---|---|---:|---:|---:|---:|
| Skip | Native | 0.6390205324 | 0 | 0 | 272 |
| Random | Positional | 0.6428851187 | -0.0038645864 | -0.0039135788 | 271 |
| W | Positional | 0.6399497688 | -0.0009292364 | -0.0009733389 | 284 |
| G | Positional | 0.6355421841 | +0.0034783483 | +0.0034160497 | 279 |
| W | Neuron-coupled | 0.6390128136 | +0.0000077188 | -0.0000386747 | 272 |

Training split: 3,240 examples; calibration: 360; validation: 256.

### SpiralHard — seed 42

| Action | Gate semantics | Validation CE | Delta CE vs skip | Net value (declared cost scale) | Final width |
|---|---|---:|---:|---:|---:|
| Skip | Native | 0.6870897412 | 0 | 0 | 268 |
| Random | Positional | 0.6870633066 | +0.0000264347 | -0.0000019158 | 269 |
| W | Positional | 0.6876192987 | -0.0005295575 | -0.0005374376 | 267 |
| G | Positional | 0.6874916852 | -0.0004019439 | -0.0004362707 | 266 |
| W | Neuron-coupled | 0.6871762872 | -0.0000865459 | -0.0001043070 | 268 |

Training split: 6,480 examples; calibration: 720; validation: 256.

## Integrity observations

1. All actions came from the same complete AWN+Adam checkpoint per task,
   consuming the same pre-materialized minibatch stream and horizon.
2. Each fork used the original AWN ELBO and intrinsic learned width update.
   Width trajectories are not constrained equal between actions after the
   intervention; actual differences are retained.
3. No-op state is explicit, and the untouched AWN skip path is the comparator.
4. The audit recomputed every label's delta-risk, incremental online cost
   and net value from its corresponding observation and skip baseline.
5. The cost scalar (0.01 CE per reference second) is a declared engineering
   convention, **not** an estimated optimal cost trade-off.
6. Outputs are stored as per-dataset GitHub Actions artifacts:
   `observations.jsonl`, `labels.jsonl`, `manifest.json`, `audit.json`.

## Scientific status and next gate

This short smoke test establishes real-reference-dataset execution and label
accounting, not statistical significance, reproducibility across checkpoints,
or out-of-sample benefit. In particular:

- G has lower CE than skip on Spiral in this single run, but higher CE on
  SpiralHard; **do not call G a universal winner**.
- The apparent tiny random improvement on SpiralHard is far below a reasonable
  basis for a superiority claim; repeated seeds, streams and checkpoints
  are required.
- Do not use the untouched final test set to tune checkpoint selection,
  labels, evidence definition or the intervention policy.
- Before fitting a value predictor, collect the predeclared multi-seed,
  multi-checkpoint engineering grid, audit policy versus skip, and split
  strictly by dataset/seed/checkpoint group to prevent teacher leakage.

The **96-rollout engineering plan** is 2 tasks x 3 seeds x 4 checkpoints x
4 actions (skip, random, W, G) x 1 future stream. It is not run or validated
by this 30-observation smoke test. A later confirmatory study requires
separate, disjoint held-out tests and uncertainty estimates.
