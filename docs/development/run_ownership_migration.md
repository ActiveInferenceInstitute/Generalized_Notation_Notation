# v4 migration: identity, scientific validation and run evidence

Version 4 preserves the numbered output layout `N_module_output`. A top-level run now
holds an exclusive OS lease on its output root until workers finish and durable
evidence verifies. Use a different output root for concurrent invocations. Before logging or any
producer writes, the lease holder inspects existing output entries and rejects
external or cyclic symlinks, hardlinked regular files, and nonregular entries.
Contained symlinks and ordinary historical files remain readable. Refusal is
written to stderr so an earlier file logger cannot modify a rejected target.
The advisory lease does not prevent filesystem changes by a noncooperating process.

Model selection is resolved once into `00_pipeline_summary/run_context.json`.
Each model has an original relative path, source SHA-256, stable path-derived
model ID, and selected step set. Private per-step input views contain exact
source bytes. Duplicate filenames receive distinct artifact stems; migrate
consumers from display-name or filename joins to `model_id` and `source_sha256`.
Documentation and archived inputs are recorded as exclusions.

Steps use registry-declared source, artifact, corpus, or run scope. Aggregate
analysis and website generation dispatch once for their selected corpus.
No selected model means no source dispatch; it never implies a root fallback.
Step 23 and Step 24 read the matching `current_summary.json` snapshot. Its
`RUNNING` status describes an invocation in progress. The final summary is
accepted only after source, output-byte, and durable-manifest checks.

An explicit `pipeline.timeout.total` is a positive, finite number of seconds.
`null` allocates a total budget from selected step budgets; LLM allocation scales
at 600 seconds per selected model. Local execution timeouts remain bounded by
the shared monotonic deadline. Budget exhaustion is a failed run with a stop
reason and recorded selected/unfinished work, even if earlier steps succeeded.

Supervised workers reserve cleanup inside an absolute invocation or request
deadline: at most one second, capped at 25% of the remaining budget. Observation,
termination, reaping and pipe draining share this reserve. Standalone subprocess
calls with only a local execution timeout retain a separate one-second cleanup
allowance; pass `deadline_monotonic` for a total request budget.

Budgeted top-level `--consolidated-steps` runs use subprocesses and record
`execution_fallback_reason: hard_deadline_requires_process_boundary`. The
serial, parallel, and matrix selections share this boundary. A worker timeout
kills its process group and descendants before output ownership is released.

On macOS, descendant discovery uses the kernel's parent-filtered child query;
on Linux, it reads child lists for each thread of owned processes. Kernels without
the Linux child-list feature and other platforms explicitly report
`process_group_only`. RSS sampling reuses observed identities and may omit
unobserved children or large trees under its 10ms advisory work ceiling. Successful
cleanup verifies the direct worker and observed descendants, rather than an OS
sandbox against hostile instantaneous detachment. Denied, incomplete or expired
cleanup fails the run even when the worker's command returned zero.

The standalone `execute_step_in_process` callback API remains available for
callers requiring in-memory injection. Its thread cancellation is cooperative:
timeout receipts carry `containment: cooperative_only`, `force_killed: false`,
and `worker_stopped`. A callback that ignores cancellation may still write
after return. Use top-level orchestration for hard containment. This API is not
evidence of process cleanup and must not share a live pipeline output lease.

Durable manifests inventory only files authored by the current run. Older
files can remain on disk for inspection but are excluded from current-run
analysis, website/report collection, and durable acceptance evidence.

Direct distributed execution now requires a ready caller-owned backend. Pass a
running Dask client as `Dispatcher("dask", client=client)`, or initialize Ray in
your own supervised process before dispatch. Automatic backend startup is
reserved for the top-level pipeline's supervised child process. Initialization
failure returns one `DistributedInitializationError` receipt per submission;
the former sequential fallback is removed. Dispatcher shutdown preserves
caller-owned clients, clusters, and preexisting Ray runtimes.

Distributed waits and result transfer share the backend limit and invocation
deadline. Timeout cancellation and asynchronous close are requests; they do not
prove that remote work stopped. The pipeline supervisor verifies its owned
process cleanup before releasing the output lease. Direct callers remain
responsible for their own cluster's startup, shutdown, and remote containment.

## Artifact contracts and summary states

The frozen context is serialized as the `RunContext` dataclass: `run_id`, input
and output roots, JSON configuration, selected frameworks and steps, model
references and exclusions. Model IDs are SHA-256 prefixes of normalized relative
source paths; source hashes separately bind raw bytes. Renaming a source changes
its ID, while a content edit changes its source hash. Display names are labels.
Do not join scientific evidence using them.

Step indexes carry the same model IDs and source hashes. Analysis emits one
result per model/framework/script identity, preserving per-agent or per-factor
views. Continuous result families carry means/covariances rather than fabricated
categorical probabilities. Semantic fidelity uses `gnn_semantic_contract_v2`;
regenerate v1 ledgers because v2 includes complete parameter values.

A summary snapshot has `RUNNING` status until finalization. Final pipeline status
is `SUCCESS`, `SUCCESS_WITH_WARNINGS`, `PARTIAL_SUCCESS` or `FAILED`; partial work
is not successful acceptance. Individual backend receipts preserve successful,
failed, skipped, unsupported, timed-out and cancelled outcomes, together with
reasons and original script identities. Unsupported scientific model kinds are
separate from missing dependencies. Explicit empty selection is valid zero work.
Required unfinished work prevents success; optional enrichment skips remain
visible. A timeout or failed output verification marks the invocation failed.

## Scientific comparison admission

Step 16 keeps operational counts and per-result timing separate from numerical
agreement. Comparisons retain model IDs, source hashes, inference meaning and
numeric precision. Matching display names or dimensions cannot admit a pair.
Missing or incompatible bindings produce unavailable results with reasons;
historical artifacts remain inspectable without acquiring verified comparison status.

Categorical summaries describe validated probability traces. Gaussian summaries
use posterior means and covariances, with uncertainty derived from covariance.
Native agent and factor views retain their own identities. Absent free energy,
confidence or inference diagnostics remain absent. These descriptive metrics do
not establish extraction correctness, universal equivalence or convergence.

## Strict scientific inputs

Direct renderers, CLI and MCP reject nonfinite, negative, zero-mass or materially
unnormalized probabilities, dimension mismatches and ambiguous/inconsistent
orientation. Declared decimal-rounding tolerance has recorded transformation
provenance; it is not unconditional normalization. Gaussian blocks require
finite dimensions, symmetric positive covariance matrices and complete optional
goal/control pairs. See [the scientific contract](backend_scientific_contracts.md).
Malformed illustrative examples remain readable and parseable, but required
rendering fails until their intended numerical models are scientifically fixed.

## Resume and historical output

LLM resume is explicit. Version-two checkpoints bind each source hash, exact provider/model,
resolved configuration, effective endpoint hash, prompt definitions and response digests; changed bindings
reject prior completion. Full-request cache identity includes all messages and
request options. Earlier cache records without these bindings are not trusted. Ollama chat requires
a verified 0.35-or-later runtime and disables input truncation/context shifting.
Sources larger than the configured context remain unfinished; the pipeline
does not sample source content or replace the model.
The configured model is preflighted; there is no silent provider/model fallback.

Durable sessions record selected units, progress and artifacts. Resume must
verify current source and artifact identities before admitting prior work; a
historical file merely present in an output folder is insufficient. Move earlier
runs to separate output roots when running concurrent sessions. The top-level
lease uses OS advisory locking on the supported POSIX runtime; Windows lease
support remains unverified rather than an assumed containment guarantee.

Experimental cpomdp is excluded from default framework selection. Install the
`cpomdp` extra, select `--frameworks cpomdp` explicitly, and preserve its admission,
source/dependency and policy score receipts. Search requests over the declared
caps fail instead of silently shrinking the horizon or action set.


## Experimental THRML artifacts and configuration

Install `uv sync --extra thrml` and select `--frameworks thrml` explicitly.
The released `thrml==0.1.4` adapter uses native categorical Gibbs sampling for
fixed-action full-sequence smoothing. Its schema retains sample witnesses,
canonical parameters, source/dependency identity and measured resources. Independent
components keep their own marginals; a joint posterior requires a declared,
bounded joint composition. C/E remain retained model values; this inference
mapping performs no policy optimization. Unsupported scientific mappings remain
explicit receipts. See [THRML implementation](../gnn/implementations/thrml.md).

Pipeline configuration uses `render.backend_options.thrml`. Step 11 accepts
`--simulation-params '{"thrml":{"num_samples":512}}'`; `gnn render` accepts
`--options '{"num_samples":512}'`. The REST render request and MCP
`render_spec_to_format` accept an `options` object. Configured values, explicit
simulation parameters, then direct options determine precedence. Unknown THRML
options fail validation. Explicit limits never change silently.

A successful THRML child exit must also provide a bounded, validated native JSON
whose execution ID and current script SHA256 match this dispatch. Generic Step 12
and the direct executor use the same validator. Old or malformed results cannot
satisfy a new run. `transition_actions` indexes the T-1 transitions; it must not
be padded into a generic T-action trace. Empirical entropy describes retained
samples; unverified convergence, calibration and EFE remain unavailable.
