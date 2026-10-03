# THRML analysis implementation

`adapter.py` owns validation of categorical arrays, dimensions, sample witnesses,
predictive semantics, declared independent components and scientific limitations.
`analyzer.py` consumes that validated representation and publishes plots/JSON.
The package facade loads these helpers lazily and never imports THRML or JAX.

Keep native component identities separate and preserve source, dependency,
sampling and runtime provenance. Do not compute Gaussian uncertainty from
categorical means, label Gibbs sampling exact, fabricate free energy, or claim
convergence/calibration from a finite posterior trace. Unsupported mappings and
invalid values fail explicitly.

Tests in `tests/analysis/test_thrml_analysis.py` cover malformed probabilities,
sample/parameter corruption, unsupported claims, independent components and
readable plots. Native execution acceptance is a separate evidence plane.

[README](README.md) covers use; [SPEC](SPEC.md) defines the shared result schema.
