# Gate G1 — Native AWN action integrity

## Scope
Checkpoint-forked controls, one-hidden-layer AWN/DynamicMLP with exact
original ELBO and Adam state. No original AWN code is edited.

G1 checks the **mechanics**, not superiority, of rank interventions.
Each action runs from an independent model+optimizer+RNG checkpoint.
The source is divided into synthetic training (for warm-up/rollout),
calibration (for evidence), and disjoint validation (for risk).
No final test split is used in decisions or labels.

Actions: skip, seeded random ordering, weight-norm ordering and gradient
ordering; each non-skip treatment is tested with positional and
identity-coupled gate modes. Follow-up work may adopt separate evidence
policies but cannot collapse these controls.

## Required checks

- Identity skip has no hooks/permutations and no evidence cost.
- Actual physical ordering maps incoming rows, outgoing columns, bias,
  and **each Adam tensor** with the same bijection, preserving width-rate.
- Identity ordering remains a marked no-op and preserves weights/moments.
- Positional random action genuinely changes predictions immediately;
  coupled gate ranking maintains the same initial function to 1e-4.
- All treatment continuations run a real AWN ELBO optimizer step with
  aligned parameters and finite held-out predictive cross entropy.
- Matched pilot has equal optimizer steps, examples, horizon, checkpoint
  and rollout seed per action, with causal label defined as:
      delta_risk = risk_skip - risk_action
  No positive advantage is presumed.
- Coupled-mode continuation must remain close to the skip baseline.
- Validations are performed on two independent synthetic seeds on CPU.

## Commands

    python -m pytest -q experiments/structural_value/tests/test_native_g1.py
    python -m experiments.structural_value.scripts.verify_g1 --seed 42 --output g1.json

The GitHub Action uploads a machine-readable control report on success.
This is **engineering verification only**, not a real-dataset research
finding. Do not begin the broad 96-rollout pilot until G1 and its audit pass.
