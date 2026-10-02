# Render frameworks

Step 11 generates code through modular backend renderers; Step 12 supervises
requested scripts and records execution outcomes. Keep the thin numbered
orchestrators, canonical `gnn.*` imports, and shared configuration contracts.

## Maintained inventory

| Framework | Language | Continuous support | Runtime requirement |
| --- | --- | --- | --- |
| PyMDP | Python | Unsupported | Declared base dependency |
| RxInfer.jl | Julia | Native | Julia and the committed RxInfer project |
| ActiveInference.jl | Julia | Unsupported | Julia and its committed project |
| JAX | Python | Native | Declared base dependency |
| DisCoPy | Python | Unsupported | Declared base dependency |
| PyTorch | Python | Native | `torch` extra |
| NumPyro | Python | Native | Declared base dependency |
| Stan | Stan/Python driver | Native | `stan` extra and explicit CmdStan installation |
| bnlearn | Python | Unsupported | `bnlearn` extra |
| cpomdp | Python | Continuous only | `cpomdp` extra; explicit experimental selection |
| THRML | Python | Unsupported | `thrml` extra; explicit experimental selection |
| ngc-learn | Python | Continuous only | `ngclearn` extra; Python 3.12 or later |

Derive inventory and help from `gnn.render.framework_registry.FRAMEWORK_REGISTRY`
and `gnn.frameworks.ALL_FRAMEWORKS`. Every renderer above has an executor. The
execution-only `lean` bridge is additional and has no Step 11 renderer. Registry
family flags describe baseline dispatch; they do not establish support for every
multi-agent, factored, hybrid, learning, or nonstationary composition.

The automatic/default and `lite` selections exclude cpomdp and THRML. Installing
an extra does not select its experimental backend. Use the backend guide for
admitted compositions and call `gnn doctor` for structured readiness diagnoses.
Missing packages, missing toolchains, unsupported Python/version, probe timeout,
probe failure, and missing executors remain distinct. Installation is explicit.

## Scientific contracts

Use the shared model-kind dispatch and extraction boundaries. Categorical models
require finite, nonnegative probabilities, valid mass, declared dimensions, and
explicit transition orientation. Canonical B is `[next_state, previous_state,
action]`. Preserve per-factor and per-agent source axes and parameter custody
when constructing a composed model; a global axis declaration cannot overwrite
unchanged component tensors. Strict paths reject malformed source probabilities.
Compatibility/demo transformations require explicit policy and provenance.

Continuous models retain `F/H/Q/R`, prior means/covariances and declared controls;
they are never replaced with categorical A/B/C/D defaults. Continuous uncertainty
comes from covariance. Unsupported quantities remain absent with a reason.
Numeric literals must round-trip, including tiny probabilities. Semantic fidelity
binds values and relevant semantics; hashes, shapes, numerical witnesses and Lean
evidence support separate claims.

THRML maps admitted positive categorical models to native factor sampling,
fixed-action Gibbs smoothing, retained samples and empirical witnesses. Structural
zeros affected by upstream issue #72 are rejected. Do not claim action
optimization, convergence, hardware operation, or energy savings. cpomdp maps
admitted linear-Gaussian models to released constructors, filtering and explicitly
requested control; validate policy admission before enumeration.

## Execution and artifact ownership

Resolve configuration once, freeze selected model identities and source hashes,
and use the selected manifest for render and execute. An explicit empty selection
means no work. Source-relative model IDs distinguish duplicate stems; display
names remain separate. Current-run render manifests select Step 12 scripts, and
current-run result indexes select analysis. Historical files cannot become new
verified evidence through directory discovery alone.

Keep hard deadlines across readiness, dispatch, result retrieval and cleanup.
Use supervised processes and record verified descendant cleanup and stream
completion. Failed cleanup, exhausted work, cancellation and incomplete required
artifacts cannot produce success. Preserve successful distributed siblings and
submission order.

Output is rooted under `output/11_render_output/<artifact_stem>/<framework>/`;
the artifact stem carries the stable model identity. Execution/analysis receipts
carry that identity, source/script hashes and invocation identity. Build the
website once from verified current-run artifacts.

## Verification and references

Test direct renderers, installed public composition, CLI, API and MCP against the
same contracts. Cover malformed probabilities, asymmetric and equal-sized B
axes, tiny values, empty/nested/duplicate selections, stale artifacts, real
children, cleanup failure and configuration precedence. Optional native acceptance
uses provisioned environments and ordinary installed wheels outside the checkout.
Keep local, hosted, numerical and native-proof receipts separate.

- [Implementation guides](../docs/gnn/implementations/README.md)
- [THRML](../docs/gnn/implementations/thrml.md)
- [cpomdp](../docs/gnn/implementations/cpomdp.md)
- [v4 migration](../docs/development/run_ownership_migration.md)
- [Verification ledger](../SCOPE-2026-10-01.md)

**Last updated**: 2026-10-02
