# Code and documentation quality

Preserve public behavior and explicit failures. Each concept has one coherent
owner with typed public interfaces and docstrings for inputs, outputs, errors,
units and constraints. Split responsibilities when justified, not solely for
line-count relief.

Use repository tool configuration and actual CI command scopes;
[pyproject.toml](../pyproject.toml) declares tools and mypy configuration.
After a frozen development install, use actual gates:

```bash
uv run --frozen --no-sync ruff format --check src scripts
uv run --frozen --no-sync ruff check src/gnn scripts
uv run --frozen --no-sync mypy src/gnn --show-error-codes
uv run --frozen --no-sync lint-imports
uv run --frozen --no-sync python scripts/check_thin_orchestrators.py
uv run --frozen --no-sync python scripts/check_flag_parity.py
uv run --frozen --no-sync python scripts/check_dep_hygiene.py
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
```

Terminology, capabilities, validation, MCP health and manuscript gates remain
required as declared in the [workflows](../.github/workflows/README.md).
Do not increase ratchets or weaken gates for a regression.

Examples use real exports and declared environments. Derive inventories from
live owners; distinguish discovery, selection, readiness and native acceptance.
Use measured test/coverage evidence instead of fixed counts. Performance,
memory, convergence, platform support and scientific equivalence need direct
evidence for their respective claims.

Review consumer/negative tests, changed paths, import pressure and public
interfaces before committing. Scratch stays outside source. Preserve
[root custody](../AGENTS.md) and
[companion pairing](../docs/development/fep_lean_paired_revision.md).

## Related contracts

[Documentation ownership and complete audits](documentation.md) owns the cross-cutting guidance.
