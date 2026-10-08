# GNN development rules

Start with [root AGENTS](../AGENTS.md), [SPEC](../SPEC.md) and the closest guide
beside affected code. Then load relevant topics below. Source, tests and CI
define executable behavior; these rules supplement owners and user instructions.

## Start here

1. Inspect checkout, full revision, dirty paths and active work.
2. Read [workflow.md](workflow.md) and the affected topic.
3. Resolve dependencies, capabilities and gates from this checkout.
4. Verify the public path and report exact evidence and limits.

Version belongs to [pyproject.toml](../pyproject.toml), published history to
[CHANGELOG](../CHANGELOG.md) and [VERSION_MAP](../docs/VERSION_MAP.md), future
work to [TO-DO](../TO-DO.md). Candidates and guide dates do not establish publication.

## Navigation by task

| Task | Guide |
| --- | --- |
| Take over or integrate work | [Workflow](workflow.md) |
| Pipeline routing and thin scripts | [Architecture](architecture.md) |
| Add or split packages | [Module patterns](module_patterns.md) |
| Selection, deadlines, resume and artifacts | [Run ownership](run_ownership.md) |
| Parse, validate and export models | [GNN standards](gnn_standards.md) |
| Render and execute backends | [Render frameworks](render_frameworks.md) |
| Compare results and scientific claims | [Scientific evidence](scientific_claims.md) |
| Tests, collection and coverage | [Testing](testing.md) |
| Code quality and static checks | [Quality](quality.md) |
| Failures, skips and retries | [Error handling](error_handling.md) |
| Dependencies and readiness | [Dependencies](dependencies.md) |
| MCP tools and transports | [MCP](mcp.md) |
| CLI, API, GUI and LLM adapters | [Interfaces](interfaces.md) |
| Runtime measurements and optimization | [Performance](performance.md) |
| Faster verified pushes and exact-source gates | [CI](ci.md) |
| Prose, examples and rule maintenance | [Documentation](documentation.md) |
| Issues, alerts and security boundaries | [Security](security.md) |
| Diagnose failures and recover | [Troubleshooting](troubleshooting.md) |
| Manuscript, companion and publication custody | [Release](release.md) |

## Essential contracts

- Public imports use `gnn.*`; numbered scripts delegate to their module.
- Frozen selection/current indexes determine work. Empty selection means no work;
  historical directories and display names cannot establish fresh evidence.
- Required failures, unfinished work and failed cleanup prevent success.
  Optional absence and unsupported science retain separate reasons.
- Deadlines include observation/cleanup. Cancellation requests and cooperative
  return do not prove a stopped worker.
- Source binding, numerical witnesses, native execution and proof stay separate.
- Keep reachable producers and normal artifact commits. Apply full manuscript/
  companion rituals whenever their inputs change.

## Locked setup and verification

```bash
uv sync --frozen --extra dev --python 3.12
uv run --frozen --no-sync gnn --help
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
```

Choose declared extras for the task. Use [testing](testing.md), [CI](ci.md),
[TO-DO verification](../TO-DO.md#verification-and-execution-rules) and
[workflows](../.github/workflows/README.md) for current gates.

## Ollama LLM integration standards

[Interfaces](interfaces.md#llm-and-ollama) owns complete-context/checkpoint guidance;
[LLM AGENTS](../src/gnn/llm/AGENTS.md) owns runtime settings and defaults.

## Directory contract

[AGENTS](AGENTS.md) explains maintenance and [SPEC](SPEC.md) defines structure/
acceptance. Existing filenames stay stable. Reviewed on 2026-10-08.
