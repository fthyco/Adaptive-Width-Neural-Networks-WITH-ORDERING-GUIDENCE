# AWN x GRAPHyco: two independent evidence-sweep stages

Implementation scope: frozen-checkpoint **ranking screening** on the corrected AWN one-hidden-layer MLP. This does **not** train 996 models. It does not modify the five original AWN modules.

## The two stages

| Stage | Selection | Fusion | Planned candidates |
|---|---|---|---:|
| `--phase 1` | all 255 nonempty subsets of `W,S,A,G,J,E,H,I` | fixed `mean_percentile` (identity ranking for singleton families) | **255** |
| `--phase 2` | all 247 multi-family subsets | the three *additional* `robust_z_mean`, `geometric_percentile`, `pareto_fronts` | **741** |

Run the stages independently, in either order. They share a measured feature-cache contract, **not** each other's outputs. Joint unique total: 996 candidates. If some features are missing, candidates are explicitly `skipped_unobserved` rather than silently scored. `--require-complete` returns nonzero unless the stage completes fully.

```bash
python -m experiments.bottleneck_triggered.evidence_sweep.sweep --phase 1 --evidence output/evidence_epoch20.npz --evaluation output/validation_epoch20.npz --output output/phase1_255.json --require-complete
python -m experiments.bottleneck_triggered.evidence_sweep.sweep --phase 2 --evidence output/evidence_epoch20.npz --evaluation output/validation_epoch20.npz --output output/phase2_741.json --require-complete
pytest -q experiments/bottleneck_triggered/evidence_sweep/tests/
```

Run from the repository root; when copying into the AWN repository, install `graphyco==0.2.0` separately for feature extraction. Frozen scoring requires NumPy; exporter requires PyTorch and the GRAPHyco core package.

## Export from real AWN checkpoint

Call the exporter from your existing training/experiment runner, when the corrected AWN model and deterministic *separate* calibration / validation DataLoaders are already available:

```python
from pathlib import Path
from experiments.bottleneck_triggered.evidence_sweep.export_awn import export_checkpoint

export_checkpoint(
    model, calibration_loader, validation_loader,
    stable_ids=stable_neuron_ids,  # maintained by the training integration; not physical slots
    evidence_path=Path('output/evidence_epoch20.npz'),
    evaluation_path=Path('output/validation_epoch20.npz'),
    checkpoint_id='dataset-seed42-epoch20',
    prior_evidence_paths=[Path('output/evidence_epoch10.npz'), Path('output/evidence_epoch15.npz')],
)
```

`H` requires two *earlier, distinct chronological checkpoints* with the same stable identity map, not two consecutive batches. If `H` cannot be computed, it is absent and the relevant stage candidates will be skipped. The current BT permutation manager does not automatically maintain stable identity across reordering and changes of physical width: that adapter must be integrated before live-BT longitudinal studies. Do not label physical slots as stable IDs after a permutation.

`E` currently means a derived **neuron-to-output edge contribution Taylor proxy**, not the GRAPHyco FX per-edge gradient. `S` uses the GRAPHyco legacy structural perturbation **proxy**. `I` is actual neuron masking effect on task cross-entropy at a frozen linear output layer. These are deliberately distinguished in metadata; no causal ranking claim follows from structure alone.

The exporter preserves model training mode, does not modify parameter gradients or take optimizer steps, and verifies reconstruction of the AWN's actual position-gated logits. Evaluation cache must be validation/train-calibration-holdout, not test. The evaluator applies AWN position-dependent gates **after** candidate reordering. Outputs are frozen-checkpoint prefix AUC and accuracy/loss at fractions 0.1–1.0, not post-training performance.

## Research boundaries

- The corrected branch `fix/bt-audit-correctness-20261115` is the baseline for integration; archived report values are pre-fix and not confirmatory.
- Stage 1/2 results must be evaluated on equal saved checkpoints with the same feature tensors and validation samples.
- Rank screening is not the matched intervention or endogenous bottleneck-triggered training experiment. These must follow with paired seeds, independent test evaluation, controls, total compute and uncertainty reporting.
- Test results using a fabricated fixture only verify code/counts, **not** AWN performance.


## Automated baseline AWN checkpoint generation (new)

From the repository root with AWN dependencies, processed DATA and DATA_SPLITS
available, install `graphyco==0.2.0` and run either independent stage:

```bash
# Stage 1: 255 family combinations (one fusion)
python -m experiments.bottleneck_triggered.evidence_sweep.train_screen \
  --dataset DoubleMoon --seed 42 --epochs 30 --checkpoint-every 5 \
  --phase 1 --output RESULTS/evidence_screen/doublemoon_seed42_phase1

# Stage 2: 741 additional multi-family fusion configurations
python -m experiments.bottleneck_triggered.evidence_sweep.train_screen \
  --dataset DoubleMoon --seed 42 --epochs 30 --checkpoint-every 5 \
  --phase 2 --output RESULTS/evidence_screen/doublemoon_seed42_phase2

# One baseline training run, with separate outputs for both stages:
python -m experiments.bottleneck_triggered.evidence_sweep.train_screen \
  --dataset DoubleMoon --seed 42 --epochs 30 --checkpoint-every 5 \
  --phase both --output RESULTS/evidence_screen/doublemoon_seed42
```

Each selected checkpoint emits an evidence NPZ, a validation NPZ,
`epochXXXX_phase1.json` or `epochXXXX_phase2.json`, and the run writes
`run_manifest.json`. Set `--save-model` for model/optimizer snapshots
(only load PyTorch checkpoint files from trusted sources). The existing original
AWN classifier/ELBO training API is reused; the sweep only changes observation
and evidence ranking. No test samples are used in calibration or screening.

### Data leakage protection and sample budgets

Original `common_utils.load_dataset_splits` aliases test data as validation
when the fold lacks a validation split. This new runner **detects that alias**,
derives a deterministic 15% validation split and 10% evidence-calibration split
from the original training fold, and trains on the remainder. If the supplied
validation is independent of test, only the calibration fraction is drawn from
training. Screening validation defaults to a deterministic 256-example cap;
adjust using `--validation-limit`. The resulting training population may be
smaller than in the historical paper/BT experiments. Therefore these runs must
not be presented as a direct reproduction of the original baselines.

Temporal H requires the same active stable identity sequence at three
checkpoints. `identity.py` intercepts AWN's real width changes, preserves
surviving neuron IDs, and assigns fresh IDs to new neurons. If width changes,
the history is reset until enough comparable checkpoints exist. No missing
gradient or temporal record is imputed with zero. A stage may validly report
skipped candidates when H cannot yet be measured.

### Test and scope

```bash
pytest -q experiments/bottleneck_triggered/evidence_sweep/tests
```

The fixed-checkpoint score is not a trained intervention outcome. This runner
does **not** execute the 14,940 paired full-training runs or modify baseline
BT permutation logic. The current implementation is one-hidden-layer MLP,
single-process CPU-first; GRAPHyco GPU telemetry is not claimed verified.
