# Developer documentation

## Overview

Start with the [contribution guide](../../CONTRIBUTING.md),
[repository instructions](../../AGENTS.md) and [source architecture](../../ARCHITECTURE.md).
The installed package is `gnn`; implementation lives under `src/gnn/`, numbered
scripts delegate to shared modules, and tests live under `tests/`.

## Quick Start for Developers

Use the uv version in the [setup guide](../SETUP.md), then from the repository root:

```bash
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync gnn preflight
uv run --frozen --no-sync python -m pytest tests/docs/ -q
```

For a release candidate, `just release-preflight` provisions the frozen dev
profile and checks deterministic release prerequisites. `just quality` runs the
broader local roster. Required hosted native, platform, security, and companion
qualification run separately.

## Project Architecture

| Task | Implementation or contract owner |
|---|---|
| Find steps and their order | [Step registry](../../src/gnn/pipeline/step_registry.py), [script guide](../PIPELINE_SCRIPTS.md) |
| Change CLI behavior | [Argument parsing](../../src/gnn/utils/arguments/arg_parsing.py), [flag parity check](../../scripts/check_flag_parity.py) |
| Change model syntax | [Normative syntax](../gnn/gnn_syntax.md), [schema package](../../src/gnn/schema/) |
| Add a backend | [Integration guide](../gnn/integration/framework_integration_guide.md), [renderer registry](../../src/gnn/render/framework_registry.py) |
| Run tests | [Testing guide](../testing/README.md), [test infrastructure](../../tests/README.md) |
| Change run/process ownership | [Migration contract](run_ownership_migration.md), [filesystem boundaries](../security/filesystem_boundaries.md) |
| Refactor shared helpers | [Utils boundaries](utils_split_design.md), [thin orchestrator pattern](thin_orchestrator_pattern.md) |

## Development Workflow

Create a focused branch, change the owning module and its public documentation,
run the affected consumer tests, and run the mandatory repository checks. Follow
[workflow rules](../../.agent_rules/workflow.md) for commits and review. Keep optional
runtime absence, unsupported models, warnings and failures distinguishable in receipts.
Do not treat an import, parser success or mocked test as native scientific acceptance.

## Documentation audit tooling

After provisioning the frozen environment:

```bash
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
uv run --frozen --no-sync python scripts/check_doc_path_references.py
```

The [audit](docs_audit.py) checks relative destinations, repository escapes,
same-page and cross-file anchors, heading duplicates, document pairing and owner
claims. Registry-bound claims are judged against source declarations: render-engine
counts against `FRAMEWORK_REGISTRY`, and model-kind coverage plus frame
documentation against each frame's `state_space` declaration. The link scan
handles multiline Markdown destinations and reference definitions, including
URL-encoded paths. It excludes inline code and fenced examples from navigation.
It writes a report unless `--no-write` is
supplied; strict failures print individual diagnostics, and `--quiet` prints counts.
[Documentation review](documentation_review.md) adds executable-example and rendered-page
review. Generated and quoted historical material keeps its declared scope.

## Remaining roadmap contracts

[TO-DO.md](../../TO-DO.md) owns target versions and outstanding work. The
[implementation contracts](remaining_roadmap_contracts.md) define bounded scientific
and source-preservation acceptance cases; they are not claims of completed features.

## Release and evidence custody

Follow [release management](../releases/README.md) and the canonical
[paired-revision order](fep_lean_paired_revision.md#closeout-ordering-canonical).
Finish content before generating source-bound manuscript and companion evidence.
Preserve producer ancestry, failed receipts and exact source identities. Read
[native coverage](native_comprehensive_coverage.md) and
[installed platform acceptance](installed_platform_acceptance.md) for the actual
release gates. Record a [contributor handoff](contributor_handoff.md) with tested
commands and unresolved limits.

## Related resources

- [Documentation hub](../README.md) and [flat index](../INDEX.md)
- [Setup](../SETUP.md), [configuration](../configuration/README.md), [troubleshooting](../troubleshooting/README.md)
- [Agent guide](AGENTS.md) and [documentation specification](SPEC.md)


## Documents in this folder

- [Processor.py Facade Census](processor_facades.md)
- [output/ Tracking Boundary](output_tracking.md)
- [Released THRML Structural-Zero Verification](thrml_released_fix_acceptance.md)
