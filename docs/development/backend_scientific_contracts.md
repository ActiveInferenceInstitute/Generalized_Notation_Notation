# Backend scientific contracts

Scientific validation precedes backend dispatch. The shared discrete validators
reject negative or nonfinite probabilities, zero mass, materially invalid
normalization, and mismatched A/B/C/D/E dimensions. They permit only an absolute
probability-mass discrepancy of `1e-4` for declared decimal rounding; corrections
are recorded in matrix provenance. This tolerance does not turn arbitrary
weights into probabilities. Numeric code literals use round-trip precision at
every rank, including tiny nonzero entries.

`extract_abcd_matrices(spec)` requires complete declared matrices. Existing callers
that deliberately construct neutral demos must opt into
`extract_abcd_matrices(spec, allow_adapters=True)`; that adapter may normalize
weights and supply defaults. It is not the strict scientific render contract.
Computed joint products may be normalized after validating every source
conditional; their provenance identifies the derived product. Invalid source
factors are never repaired by that computation.

B axes must retain source meaning. `ModelParameters.b_tensor_order` can declare
`next_state_previous_state_action`, `action_next_state_previous_state`, or
`action_previous_state_next_state`. The bistable exemplar now explicitly declares
its outer action blocks as `action_next_state_previous_state`, preserving every
numeric value. Native `B_agentN` blocks accept the same declarations. A per-agent
`b_tensor_order_agentN` overrides the shared declaration; source provenance is
retained. Undeclared earlier native agent blocks use
`(action, next_state, previous_state)`. The compiler applies the declared
permutation without transposing conditional next/previous meaning. Native
`E_agentN` habit priors are explicitly unsupported, rather than omitted. Native
agent projection avoids allocating unused joint
tensors, and scripts embed native parameters and a source-contract digest.

The continuous contract requires positive dimensions and timesteps, finite
system matrices and prior means, positive finite `dt`, and symmetric positive
definite process, observation, and prior covariances. Current native programs use
nonsingular Gaussian densities and Cholesky sampling. Degenerate Gaussian models
need a separate declared capability. Optional `goal_mean` and `control_gain` must
both be present, dimensionally valid, and finite. Each factored or agent block
uses the same validator.

## Independent continuous agents

`continuous/independent_gaussian_agents.md` declares asymmetric one- and
two-dimensional agents. To render this composition in JAX or RxInfer, declare
`agent_coupling: independent`, `nr_agents: N`, and complete
`F_agentN/H_agentN/Q_agentN/R_agentN/prior_mean_agentN/prior_cov_agentN` blocks for
each contiguous agent identity. Optional goal/gain pairs are per agent. Declare
`random_seed_agentN` to bind individual noise streams; otherwise seeds are
`random_seed + N - 1`.

The `gnn_multi_agent_continuous_v1` result retains per-agent means, posterior
covariances, controls, observations, truth, and validation. It contains no
fabricated joint categorical state space. A shared flat Gaussian block plus an
agent count remains unsupported: independent filters must not be inferred from
that ambiguous declaration. JAX reports online filtering and RxInfer reports
batch smoothing; they estimate different posteriors and need not agree
numerically.

## Result analysis and semantic fidelity

`analysis.result_adapter` validates regular finite trace shapes, categorical row
mass, native agent/factor identities, Gaussian covariance symmetry/PSD, and
reported timestep alignment. `result_views` retains separate coherent native
marginals. It does not invent a joint posterior, substitute a first factor, or
fill missing truth/observations with zero-valued trajectories.

PNG, GIF, and self-contained HTML use the same result views. Gaussian plots show
means with 95% marginal Gaussian intervals; inference VFE is indexed by inference
iteration. Discrete native agent/factor views remain separately labeled.
`continuous_result_metrics` returns covariance checks, posterior standard
deviation, and RMSE only when aligned truth is actually reported.

`gnn_semantic_contract_v2` includes complete parameter values and declared
parameter metadata as well as names, edges, dimensions, shapes, equations,
ontology, timing, and flags. Same-shape parameter changes alter the contract and
fail comparisons. Regenerate old semantic ledgers to migrate from v1; a hash
match remains an identity claim, not evidence that a backend implements the
model or that different inference estimands are equivalent.

## Readiness, setup, and bounded probes

`check_framework` probes the actual requested interpreter in a subprocess
with a 30-second default deadline, clamped to a pipeline invocation's remaining
time. Termination, reaping, and inherited-pipe draining share one additional
second through the supervised subprocess envelope. Results are reused only
within that invocation and interpreter. A failed cleanup is `probe_failed`,
retaining the original timeout under `execution_error_type`; cleanup evidence
describes observed descendants, not containment of arbitrary detached children.
`FrameworkStatus.reason_code` distinguishes `missing_module`,
`missing_toolchain`, `unsupported_python`, `unsupported_version`, `probe_timeout`,
`probe_failed`, and `executor_unavailable`. Timeout, import failure, or interpreter
failure does not establish package absence. Doctor, execution planning, and
Step 12 receipts retain these reasons; install hints apply to established
installation or version gaps. cpomdp requires the released `0.4.4` distribution. Named Julia backends probe
all required packages in the committed project (Julia >= 1.10) with startup files
disabled and Pkg offline mode. Doctor separates launcher PATH evidence from
`kind: julia_project` package readiness and reports runtime/backend versions.
The planner's `ready` status records discovered candidates; Doctor's aggregate
`execution_ready` is true only when `would_execute` contains a runnable script.
An all-skipped plan therefore reports `execution_ready: false`.

Step 1 recognizes explicit `torch`, `stan`, `bnlearn`, `ngclearn`, `cpomdp`, and `thrml`
extras. ngc-learn requires Python 3.12 or newer. Stan's Python driver and its
CmdStan compiler installation are separate prerequisites. See
[Stan installation](../gnn/implementations/stan.md#installation); installing an
extra does not download CmdStan automatically. Readiness probes never install or
download toolchains.

## Focused verification

```bash
uv run --frozen --no-sync python -m pytest \
  tests/render/test_probability_precision.py \
  tests/render/test_continuous_validation.py \
  tests/render/test_independent_gaussian_agents.py \
  tests/analysis/test_scientific_result_adapter.py \
  tests/utils/test_framework_probe_reasons.py \
  tests/pipeline/test_semantic_fidelity_gate.py -q
```

The explicit Julia test uses the `needs_julia_env` toolchain marker. Live receipts
must record source/script digests, declared timestep or iteration overrides,
backend versions, validation, and resulting artifact inventory separately from
unit-test results.

The precision-weighted exemplar's explicitly equal three-state prior and habit
now use `0.3333333333333333` for each entry. Its former `0.333` entries had total
mass `0.999`, outside the admitted rounding tolerance. The source comments record
this explicit uniform-intent correction; regression tests still reject the
original distribution.

The HMM baseline and T-maze source values remain unchanged pending scientific correction. HMM emissions have column mass 1.5; T-maze transitions have zero-mass previous-state columns. Their parser coverage remains enabled, and required rendering returns failure. They have explicit invalid-source regression cases and are marked illustrative in the corpus index. Static/dynamic perception likelihood intent is also unresolved; material mass errors are rejected rather than repaired.


## THRML categorical sampling

The explicitly selected [THRML adapter](../gnn/implementations/thrml.md) requires
the released `thrml==0.1.4` distribution and verified categorical APIs under a
bounded interpreter-specific probe. It conditions on complete observations and
fixed transition actions, using native THRML Gibbs draws. Posterior smoothing
and replicate-observation prediction are distinct from online filtering and
one-step future prediction. Numerical comparison requires the same model,
observations, actions and inference interpretation; Monte Carlo tolerances and
seed/sample counts must accompany sampling evidence.

The result validator checks canonical probabilities/dimensions, retained sample
counts/indices, empirical frequencies and replicate prediction against A. It
retains source semantics and separate component marginals. Resource admission
estimates tensor allocation before composition and sampling; process RSS includes
JAX runtime/compilation and is recorded separately. Estimated allocation is not
measured peak RSS. Unsupported model families or structural-zero mappings are
reported explicitly, without clipping or silently normalizing their values.
