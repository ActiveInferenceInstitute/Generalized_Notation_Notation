# Behavior testing

Read [tests/AGENTS](../tests/AGENTS.md) and the relevant consumer suite. Import
the installed `gnn.*` package; do not add alternate top-level module paths,
manipulate `sys.path` in new tests or skip missing required imports.

Use real authored models, temporary output roots, public entrypoints and
observable artifacts. Check what users depend on: empty selections write no
model artifacts, requested backends are rendered, malformed inputs fail with
useful diagnoses and cancellation stops owned work within budget. Repeating
implementation logic does not establish an independent behavior contract.

Run focused tests first. Default CI selections include:

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --no-sync python -m pytest tests/ -q --tb=short   -m 'not pipeline and not mcp and not ollama and not env_heavy and not toolchain'
uv run --frozen --no-sync python -m pytest tests/ -q --tb=short   -m 'pipeline and not ollama and not env_heavy and (not toolchain or needs_posix)'
uv run --frozen --no-sync python -m pytest tests/mcp/test_mcp_audit.py -q
```

The [CI workflow](../.github/workflows/ci.yml) defines complete functional
MCP/capability selection, Python matrix, extras and JUnit/coverage artifacts.
GUI/browser/network/audio/providers/toolchains use explicitly provisioned
acceptance. Missing optional distributions and broken installed imports are
different outcomes; required failures must fail.

Retain per-environment reports, source identity, failures and skips. Overlapping
selections are not a unique total. The [coverage roadmap](../TO-DO.md#medium-work)
targets more than 80% statements on its declared matrix. Measure and qualify
optional surfaces before changing the enforced floor; preserve existing scope.
