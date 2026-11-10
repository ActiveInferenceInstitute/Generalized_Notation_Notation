# Generalized Notation Notation (GNN) Documentation

This is the primary human-facing documentation hub. Other navigation files have
narrow roles:

- [START_HERE.md](START_HERE.md) is a short audience map.
- [INDEX.md](INDEX.md) is a flat link index.
- [learning_paths.md](learning_paths.md) contains the curricula.
- [CROSS_REFERENCE_INDEX.md](CROSS_REFERENCE_INDEX.md) maps topics and frameworks.
- [AGENTS.md](AGENTS.md) describes the documentation tree for maintainers.

The code and installed CLI define behavior. Documentation gates check examples
and claims at the scopes described below. The [version map](VERSION_MAP.md)
records release and distribution identities.

## Choose a path

### New to GNN

1. [What is GNN?](gnn/about_gnn.md)
2. [Quickstart tutorial](gnn/tutorials/quickstart_tutorial.md)
3. [Syntax reference](gnn/reference/gnn_syntax.md)
4. [Examples](gnn/tutorials/gnn_examples_doc.md)

### Running the repository

1. [Setup](SETUP.md)
2. [Configuration](configuration/README.md)
3. [Pipeline guide](pipeline/README.md)
4. [Troubleshooting](troubleshooting/README.md)
5. [Framework availability](execution/FRAMEWORK_AVAILABILITY.md)
6. [Dashboard](gnn/modules/20_website.md) — the run's Step 20 website at `output/20_website_output/index.html`

### Developing or integrating

1. [Pipeline architecture](../src/gnn/README.md)
2. [Pipeline scripts](PIPELINE_SCRIPTS.md)
3. [GNN implementation guide](gnn/integration/gnn_implementation.md)
4. [Framework integration](gnn/integration/framework_integration_guide.md)
5. [API documentation](api/README.md)
6. [Development guide](development/README.md)

### Research and theory

- [GNN overview](gnn/gnn_overview.md)
- [GNN paper and formal background](gnn/gnn_paper.md)
- [Active Inference foundations](active_inference/README.md)
- [Cognitive phenomena](cognitive_phenomena/README.md)
- [Advanced modeling patterns](gnn/advanced/advanced_modeling_patterns.md)

## Basic Examples

- [Quickstart tutorial](gnn/tutorials/quickstart_tutorial.md)
- [Example gallery](gnn/tutorials/gnn_examples_doc.md)
- [Basic template](templates/basic_gnn_template.md)

## Framework Integrations

See the [framework integration guide](gnn/integration/framework_integration_guide.md)
and [framework implementations](gnn/implementations/README.md). For the Lean 4 frontier, see the
[FEP_Formal bridge program](other/fep_lean/README.md) (contract v0.7; canonical
contract in the sibling checkout).
For spatial active inference, see the [GEO_INFER interchange](other/geo_infer/README.md)
(contracts `gnn-geo-infer/1|2|factored/1`; canonical interchange docs in the sibling checkout).

## Who already uses GNN

The [downstream registry](downstream/README.md) lists the repositories that apply
GNN, each with its specification path, pinned GNN revision and conformance
evidence. Each row records the revision and verification scope.

## Current implementation map

The main pipeline has 25 numbered steps, 0–24. The canonical order and descriptions
live in `src/gnn/pipeline/step_registry.py`; the generated/maintainer-facing tables are
in [src/gnn/STEP_INDEX.md](../src/gnn/STEP_INDEX.md) and [PIPELINE_SCRIPTS.md](PIPELINE_SCRIPTS.md).

Framework boundaries are deliberately explicit:

- Step 11's [renderer registry](../src/gnn/render/framework_registry.py) owns
  code generators, model-kind support and default/experimental selection.
- Step 12's [direct executor specifications](../src/gnn/execute/executor_specs.py)
  and [pipeline discovery](../src/gnn/execute/detection.py) have distinct
  inventories. Lean document checking is separate from numerical simulation.
- Experimental cpomdp and THRML require explicit selection. Composition admission
  remains backend-specific; a registered generator does not imply that every
  discrete, continuous or coupled model can run.
- Missing optional runtimes retain skipped/unavailable outcomes. Stan needs its
  cmdstanpy driver and CmdStan; Julia backends use committed project environments.
  See [framework availability](execution/FRAMEWORK_AVAILABILITY.md) and
  [pipeline boundaries](pipeline/README.md#step-boundaries).

Read [Framework Implementations](gnn/implementations/README.md) for per-target
material and [the framework integration guide](gnn/integration/framework_integration_guide.md)
for the render → execute → analyze contract.

## Command reference

All commands below run from the repository root:

```bash
# Install and inspect the environment.
uv sync --frozen --extra dev --python 3.14
uv run --frozen --no-sync gnn preflight
uv run --frozen --no-sync gnn health

# Validate one file.
uv run --frozen --no-sync gnn validate input/gnn_files/discrete/two_state_bistable.md --strict

# Run a focused pipeline path.
uv run --frozen --no-sync gnn run \
  --target-dir input/gnn_files/basics \
  --output-dir output/docs-quickstart \
  --only-steps 3 5 6 7 8 \
  --skip-steps 0 1 2
```

The pipeline loads `input/config.yaml`. Use `gnn run --help` for step selection,
worker and logging options, and the [configuration guide](configuration/README.md)
for configuration precedence and backend settings.

## Documentation quality checks

```bash
uv run --frozen --no-sync python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --frozen --no-sync python scripts/check_doc_contracts.py --strict
uv run --frozen --no-sync python scripts/check_gnn_doc_patterns.py --strict
uv run --frozen --no-sync python scripts/check_maintained_doc_terms.py --strict
```

The audit checks links, anchors, and documentation scaffolding. The contract check
covers the executable quickstart fixture, supported CLI spellings, configuration
location, and framework split. It does not replace running a real pipeline.

## Inbound navigation anchors

These headings preserve inbound links from older topic pages while the navigation
hubs are consolidated above.

### Basic Examples

See [Basic Examples](gnn/tutorials/gnn_examples_doc.md).

### Framework Integrations

See [Framework Integrations](gnn/integration/framework_integration_guide.md).

## Maintainer notes

- Prefer links to source-backed references over duplicated inventories.
- Do not embed test counts, file counts, byte sizes, or “production-ready” claims
  unless the measurement is generated by a checked-in command and scoped to a dated
  run.
- Keep generated pipeline artifacts under `output/`; do not treat them as maintained
  documentation.
- Keep the GNN syntax examples aligned with `src/gnn/schema/parser.py` and validate examples
  with `gnn validate` or Step 5 before publishing them.

For versioning policy, see [SPEC.md](SPEC.md). For contribution conventions, see
[style_guide.md](style_guide.md) and [../CONTRIBUTING.md](../CONTRIBUTING.md).
