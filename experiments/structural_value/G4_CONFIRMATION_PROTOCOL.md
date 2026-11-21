# G4 — Precommitted New-Seed Prospective Confirmation

## Locked source

G3 model is loaded **without refitting** from the successful G3
artifacts (Actions run `44849364139`, artifact `14683998134`).
It is the exact full-G2-data Ridge candidate previously exported by G3.
Fixed alpha=12, training-only G2 normalization and coefficients, nine
AWN predecision features + dataset flag, zero intervention threshold,
SKIP priority on ties. No G4 data are used to fit the predictor.

## Predeclared G4 design

| Dimension | Locked value |
|---|---|
| Datasets | Spiral, SpiralHard |
| Seeds, absent from G2/G3 | 20271, 20272, 20273 |
| Checkpoint epochs | 5, 10, 20, 35 |
| Actions | native AWN SKIP, random positional, W positional, G positional |
| Future rollouts | same eight training-only optimizer steps per action |
| Observed horizons | 1, 4, 8 steps |
| Predictive score | future CE risk gain − 0.01 × incremental online seconds |
| Baselines | always SKIP, Random, W, G; hindsight oracle **diagnostic only** |
| Final test set | never used in feature, teacher label or policy selection |

New seed trajectories constitute **24 fresh decision contexts** and
a **96-fork prospective confirmation** (24 SKIP, 72 interventions)
with 288 horizon observations. Four checkpoints per one seed share
training history; they are not independent replicas.

## Strict temporal separation

GitHub Actions enforces a barrier between two entirely separate job
matrices:

**Stage A — Predictions precommitted**

1. Download G3 model artifact from the fixed historical run ID.
2. Validate model class, feature/target order, alpha, training population,
   source SHA and non-use of held-out test labels.
3. Train a single native AWN baseline per dataset and new seed; record
   *pre-intervention* checkpoint features.
4. Save full native AWN model+Adam+RNG snapshots at four epochs, and save
   prediction, action, frozen model hash, snapshot content hash in
   `precommit.json`.
5. Publish the prediction artifact. This program never forks the model
   for counterfactual future labels and cannot inspect outcomes.

**Global barrier:** No Stage B job can run until all six Stage A jobs pass.

**Stage B — Outcomes obtained after prediction commit**

1. Download exactly the corresponding Stage A artifact and source snapshot.
2. Validate source SHA, model hash, frozen action, numerical predictions,
   checkpoint identity and SHA-256 of all four snapshots *before*
   running the first intervention.
3. Recreate the reference train/calibration/development-validation splits.
4. Run original AWN + unchanged ELBO and match SKIP/Random/W/G
   independent post-checkpoint forks on the same eight-step minibatch stream.
5. Recompute every label and cost equation, no-op flags and training
   budget via the independent pilot audit. Preserve raw output JSONL
   and immutable prediction provenance.

**Stage C — Fail-closed aggregate**

Download all six Stage B artifacts; recompute all 24 checkpoint audits
and verify predictions, interventions, source G3/commit and snapshot
fingerprints. Abort if any seed, decision or treatment is missing.

Predeclared primary comparison:
mean new-seed H8 incremental net value of the *locked Ridge policy*
minus native SKIP (zero). Secondary: CE-only gain (no wall time),
harmful/beneficial decisions, worst loss, per seed and dataset,
and comparisons with always-Random/W/G.

## Scope and limitations

The experiment checks generalization to *new training seeds* on the
same two original reference datasets. Even if positive, it cannot
establish robustness on new task families, hardware-stable cost
savings, or significant production performance. Timings at
millisecond scale remain noisy and the horizon is only eight steps.
No final test labels are used and no hyperparameter modification
is allowed after inspecting G4 outcomes. A failure is reported
as a failure, not fixed by selecting better G4 seeds or thresholds.
