# THRML execution contract

The lazy public API exports `execute_thrml_script`, `find_thrml_scripts`,
`is_thrml_available`, `run_thrml_records`, `run_thrml_scripts`, and
`validate_native_result(payload, execution_id=..., script_sha256=...)`.

The shared availability diagnosis checks the exact released `thrml==0.1.4`
distribution, required categorical APIs and a working JAX device in the selected
Python interpreter. The wheel requires Python >=3.10; GNN's supported lock
splits and dependency acceptance remain the repository's authority. Probes are
bounded, cancellable process groups and never install dependencies.

Scripts are security-gated before readiness, then executed by the shared safe
executor. One finite positive budget intersects the invocation and pipeline
deadlines. Known cleanup failure takes priority over cancellation and timeout.
The output lease covers execution, validation and receipt publication.

Native JSON is limited to 16 MiB and must satisfy the analysis adapter's schema,
probability dimensions, retained sample witness and replicate prediction.
The emitted runtime execution ID and script SHA256 must match the caller's
binding. Successful child exit alone cannot verify stale or malformed results.
Late required validation or receipt publication makes the outcome unsuccessful
while preserving stdout, stderr, scientific artifacts and known cleanup facts.

Batch discovery is deterministic and deduplicated. Relative source paths give
duplicate stems distinct output identities. Successful siblings remain available
when another selected script fails; nonempty required skips never imply success.

The backend supports verified positive-probability discrete categorical
smoothing and explicitly independent components. Structural zeros, unsupported
couplings and continuous mappings require separate verification before use.
