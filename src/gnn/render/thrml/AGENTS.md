# THRML renderer agent

This module owns the experimental `thrml` rendering boundary. Select THRML
explicitly; it is excluded from default backend selection. The released
dependency is `thrml==0.1.4`, installed with `uv sync --extra thrml`.

`render_gnn_to_thrml(spec, output_path, options)` returns the standard
`(success, message, artifacts)` tuple. It renders without importing THRML,
JAX, or Equinox. `runtime.py` imports native packages only after serialized
value, semantics, dimension, and aggregate admission checks pass. Numbered
Step 11 remains a thin orchestrator.

Preserve canonical B axes `(next_state, previous_state, action)`, all source
parameters, preferences/habits, source structure, and semantic hashes.
Native transitions require a declared orientation. Independent component
groups stay separate; declared cross-component coupling must not disappear.
Dependent factors use the bounded shared exact categorical composer. Modal
likelihood products retain their source tables and Cartesian ordering.

The implemented inference conditions on fixed transition actions and an
entire finite observation sequence. Actual `FactorSamplingProgram` and
`sample_states` perform categorical block Gibbs draws. Empirical marginals
are approximate full-sequence smoothing estimates. Preserve the distinction
between exact source composition, a parameter hash, Monte Carlo accuracy,
convergence, and action-control semantics.

Strictly positive finite A/B/D probabilities are required by the verified
stock-sampler mapping. Structural zeros, continuous/nonstationary dynamics,
and unrepresented coupling produce explicit unsupported receipts. Do not
clip zeros, repair malformed mass, substitute a sampler, synthesize free
energy, or claim THRML hardware acceleration from local JAX execution.

Admit the complete component set and retained source matrices before array
copies or Cartesian/native allocation. Current caps are 16 components,
4,096 categories per categorical variable, 1,048,576 native factor entries
and retained matrix entries, 4,194,304 site updates, and a 128 MiB allocation
estimate. The estimate is not an upper bound; the process watchdog is the
authoritative runtime limit.

Verification commands:

```bash
uv run --extra dev python -m pytest tests/render/test_thrml_contracts.py tests/render/test_thrml_surfaces.py -q
uv run --extra dev --extra thrml python -m pytest tests/render/test_thrml_native.py -q
```

Native tests must actually execute the released wheel and compare its
sample-derived marginals against independent log-space forward/backward
math. Use both supported JAX lock splits before release. Keep missing-package
checks distinct from genuine native acceptance.

See [README](README.md), [SPEC](SPEC.md), and the
[THRML implementation guide](../../../../docs/gnn/implementations/thrml.md).
