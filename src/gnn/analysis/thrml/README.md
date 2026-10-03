# THRML analysis

`adapt_result(payload)` validates the experimental discrete result schema.
`analyze_payload(payload, output_dir)` writes readable posterior/predictive PNGs
and `thrml_analysis.json`. `generate_analysis_from_logs` analyzes each selected
native result once; pipeline scheduling supplies current-run artifacts.
Managed pipeline, maintained batch, and direct execution destinations share
native result discovery. Foreign framework directories are excluded. Explicit
model IDs remain authoritative; when absent, a safe relative-path identity
keeps scripts with equal stems in different directories distinct.

The posterior is a Gibbs Monte Carlo estimate of the full observation sequence
under fixed transition actions. Retained `sample_states[sample, timestep]`
must reproduce the reported `beliefs[timestep, state]`. Predictive observations
describe a replicate draw conditional on those smoothed states, not held-out
forecast performance. Named independent components retain separate marginals
and sample traces rather than a fabricated joint posterior.

Empirical entropy and maximum probability describe the reported sample
distribution. Variational/expected free energy, convergence, effective sample
size and calibration remain unavailable with reasons. Gaussian covariance and
continuous models are unsupported. Model, source, dependency and runtime
identities remain attached to the analysis.

[Execution](../../execute/thrml/README.md), [specification](SPEC.md), and
[implementation guide](../../../../docs/gnn/implementations/thrml.md) provide the
selection and scientific contract.
