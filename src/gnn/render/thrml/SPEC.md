# THRML renderer specification

The renderer API is `render_gnn_to_thrml(spec, output_path, options=None)`.
It writes one Python script atomically after successful admission. Rejected
semantics return `(False, "unsupported-thrml: ...", [])`; malformed source,
options, or filesystem failures return a failed render with no new script.
Existing files are not success evidence for a rejected current render.

Options are `num_timesteps`, `num_samples`, `burn_in`, `thin`, `seed`,
`observations`, and `transition_actions`. Every count is an actual integer;
Booleans, strings, fractional counts, and invalid ranges are rejected.
Observations contain `T` indices, actions contain `T-1` indices. The seed
lies in `[0, 2**32-1]`. Effective options merge configured
`backend_options.thrml`, command `simulation_params.thrml`, common `timesteps`,
and direct backend options in that precedence order. Unknown options fail.

The emitted JSON payload has `schema_version: 1`, `framework: thrml`,
`model_kind: discrete`, `experimental: true`, a semantic SHA-256 binding,
source model parameters, components, sampling schedule, float64 sampling
policy, admission counts, and explicitly named inference/predictive/action
semantics. Each component retains canonical and original parameters, matrix
transformation provenance, source structure, and composition descriptors.
`source_model_parameters` preserves the original input declarations;
`composed_model_parameters` records canonical component dimensions and axes
after source composition. Derived joint dimensions must not replace a field
labelled as original source metadata.

Canonical source validation requires finite probability mass and dimensions
consistent with A/B/C/D[/E]. A/B/D must have strictly positive support for
the verified stock sampler. Canonical B uses `(next, previous, action)`.
Declared source axis aliases are canonicalized before action selection.
Decimal precision corrections remain bounded by the shared canonical
contract and carry source/transformation provenance; malformed mass is not
unconditionally normalized.

Independent complete A/B/C/D component groups are sampled separately.
Connections across components are unsupported. Dependent factor/modal source
contracts are composed only after source dimensions and every action's
retained transition storage pass admission. Exact categorical composition
does not promote the subsequent empirical Gibbs posterior to exact inference.
Source kind predicates reject continuous, hybrid, nonstationary, learning,
hierarchical, structural, and unrepresented coupled semantics.

For each component with states S, observations O, duration T, retained samples
K, burn-in B, and thinning L:

```text
factor_entries = S + max(T-1,0)*S*S + T*O*S
site_updates = (B + K*L)*T
# Synthetic joint draw adds (B+L)*2*T updates.
retained_matrix_entries = sum(size(A), size(B), size(C), size(D), size(E if present))
estimated_bytes = 64*(factor_entries + retained_matrix_entries)
                  + 32*K*T + 256*T*(S+O)
```

Caps apply to the aggregate across all components: 1,048,576 factor entries,
1,048,576 retained matrix entries, 4,194,304 site updates, and 128 MiB estimated
allocation; components are capped at 16 and categorical widths at 4,096.
The estimate is explicitly not an upper bound. JIT/import overhead and
actual process memory remain subject to the executor watchdog.

Runtime revalidates the complete payload and aggregate admission before
native package imports. Released THRML 0.1.4 `FactorSamplingProgram`,
`CategoricalEBMFactor`, `CategoricalGibbsConditional`, and `sample_states`
perform sampling. Even/odd state blocks respect chain factor dependencies;
observation nodes are clamped for posterior sampling. JAX x64 is enabled
before numeric factors are constructed, preserving tiny positive literals.

Results retain `sample_states[K,T]`, `beliefs[T,S]`, observations,
`transition_actions[T-1]`, replicate predictive distributions, composition
marginals, source values and semantics, dependency versions, execution token,
script SHA-256, elapsed time, allocation estimate, and measured process peak
RSS where available. Independent results have a `components` mapping and no
fabricated joint belief array. No VFE/EFE, calibration, convergence, or
hardware-performance claims are synthesized.

See [README](README.md) and [AGENTS](AGENTS.md).
