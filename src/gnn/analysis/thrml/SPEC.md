# THRML scientific result contract

Schema version 1 declares `framework=thrml`, `model_kind=discrete`,
`inference_mode=gibbs_monte_carlo`, `success=true`,
`posterior_semantics=fixed_actions_full_sequence_smoothing`, and
`predictive_semantics=replicate_observation_given_smoothed_state`.
Mapping support and conditional validity must be affirmed explicitly.
The transition action interpretation must be `transition_into_next_state`;
`sampling_dtype=float64` and the runtime's `jax_enable_x64=true` declare the
verified numerical policy. `source_identity.semantic_sha256` is a lowercase
64-character hexadecimal digest with the explicit parameter/mapping identity
claim; it cannot be relabelled as proof of extraction or Monte Carlo accuracy.

For T timesteps, S states, O observations, U actions and K retained samples:

| Field | Meaning |
| --- | --- |
| `parameters.A[O,S]`, `B[S,S,U]`, `C[O]`, `D[S]` | Declared model, with canonical next/previous/action B axes |
| `parameters.E[U]` when present | Retained source habits; fixed actions perform no habit-based policy inference |
| `sample_states[K,T]` | Integer Gibbs sample witness |
| `beliefs[T,S]` | Empirical categorical probabilities from those samples |
| `predictive_observations[T,O]` | `beliefs @ A.T`, with the declared replicate interpretation |
| `observations[T]` | Clamped observations |
| `transition_actions[T-1]` | Action at t conditions the transition into t+1 |
| `sampling` | Integral timestep count, sample count, burn-in, thinning and uint32 seed |

A/B/D must have positive finite normalized probability mass; structural zeros
are explicitly unsupported without a verified support-aware adapter. Optional E
must remain a finite normalized nonnegative source vector. No transformations
repair malformed arrays. Native component results use
`composition=independent_components` and `components[name]`; component IDs,
contract metadata and timestep counts must agree with the enclosing receipt.
The native schedule's timestep count must match the posterior length; every
independent component must retain the same effective schedule as its parent.

Joint categorical compositions retain named state-factor and observation-modality
marginals. Their source descriptor cardinalities must multiply to the joint
width; every marginal must equal the declared tensor projection of the verified
empirical posterior or replicate prediction. Analysis preserves composition,
sampling, source-structure, measured resource usage, and extensible resource-admission provenance and
plots each represented factor or modality separately. Duplicate destination
identities fail before publication rather than overwrite another model's report.
Original `source_model_parameters` remain separate from derived
`composed_model_parameters`; numerical metric dimensions come from the verified
joint trace rather than relabelling original factor or modality cardinalities.

Unsupported numerical quantities remain absent with reasons. An artifact hash
establishes identity, not extraction correctness, MCMC convergence, calibration,
deterministic equivalence or a formal Lean claim.
