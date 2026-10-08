# Module ownership and interfaces

Each module under `src/gnn/` owns one coherent concept. Public entrypoints live
in its `__init__.py`; processing, native backends and MCP wrappers delegate to
named implementation owners. Read the module's `AGENTS.md`, `README.md` and
`SPEC.md` before changing its contract.

Preserve supported public exports and signatures. The existing lazy facades
and import-boundary tests govern package isolation; do not copy a new export
map or advertise static feature metadata as verified native readiness.
Split overloaded responsibilities when import pressure and consumer behavior
justify the boundary, not solely to make files shorter.

Processing functions use actual module signatures. Shared numbered-script
setup belongs to the
[script factory](../src/gnn/utils/pipeline_orchestration/pipeline_template.py).
Use the existing run context, selected manifest and current artifact indexes
instead of adding a local result type or rediscovering arbitrary directories.

MCP schemas and wrappers preserve requested arguments, typed validation,
readiness and real outcomes. Returning a dictionary around a failed operation
does not make it successful. See [MCP guidance](mcp.md).

Keep optional/native imports at their owning boundaries. Test real public
consumers, failures and import pressure; document the ownership invariant and
working usage. Scratch probes stay outside the source tree. See
[quality](quality.md) and [testing](testing.md).

## Related contracts

[Task ownership, integration and closeout](workflow.md) owns the cross-cutting guidance.
