# Gate G0 — Native AWN integrity

G0 means an unchanged AWN run and the skip adapter must produce identical
parameters, optimizer tensors, widths, rates, validation logits and torch
RNG states after every matched optimizer step from the same checkpoint.
Two independent skip forks must also agree.

## Mandatory run

Use the official AWN Python environment with mlwiz, PyTorch and PyG.

    python -m pytest -q experiments/structural_value/tests
    python -m experiments.structural_value.scripts.verify_g0 --output g0-native-report.json

The native G0 runner creates synthetic training and validation examples,
not paper benchmark datasets, and never uses the final test split. It trains
the actual model.AWN and original metric.ELBO_Classification. It snapshots
a checkpoint with Adam momentum and a live autograd-linked gate cache, restores
forks, and compares original AWN vs skip and skip vs skip at every step.

The real paired engine is tested twice with skip and actual intermediate
horizons 1, 2, 4. Wall-time fluctuations are not equality assertions.

A second stress case sets the AWN exponential rate to the same sequence
in all branches, forcing physical shrink/growth (8, 13, 7, 12) and checking
Adam moment migration. This does not test a policy or claim beneficial width
adaptation. It is an explicit integrity control.

The GitHub workflow structural-value-g0.yml fails if required dependencies
or any native AWN equality assertion fails. On success, it publishes the
immutable CI run's native JSON report as a workflow artifact.

## Acceptance

- Exact CPU tensor equality for model parameters, Adam state, logits, RNG;
  equal width and width distribution parameters at every step.
- No missing/duplicated optimizer parameters after AWN resize.
- The source model's non-leaf qW cache is not accidentally detached.
- Original AWN modules remain unchanged.
- Skip neither installs physical permutation hooks nor acquires evidence.

This is an engineering gate, not evidence of new learned adaptation. GPU replay,
same-wall-time fairness, positional/coupled action effects and grouped predictors
remain separate work.
