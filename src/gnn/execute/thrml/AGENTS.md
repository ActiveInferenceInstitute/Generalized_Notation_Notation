# THRML execution implementation

Maintain `thrml_runner.py` as orchestration over shared security, availability,
process, deadline and output-lease helpers. `__init__.py` remains lazy and keeps
optional THRML/JAX imports out of the parent. Preserve structured failure causes,
script/source identity and successful siblings; never normalize invalid results
or reinterpret an unavailable runtime as a successful run.

Coordinate changes to the emitted native schema with `render/thrml` and
`analysis/thrml`. Shared planning, doctor and executor registries consume the
same availability diagnosis. Experimental selection remains explicit.

Contract tests live in `tests/execute/test_thrml_runner.py` and
`tests/analysis/test_thrml_analysis.py`. Protocol fixtures verify real child and
artifact boundaries; they do not establish native inference. Acceptance must
execute generated scripts against the ordinary released wheel on each supported
JAX lock split, with distinct receipts for those native runs.

[README](README.md) describes use and [SPEC](SPEC.md) defines the contract.
