# GNN Pipeline - Master Agent Scaffolding

## Overview

The GNN (Generalized Notation Notation) Pipeline is a comprehensive 25-step system for processing Active Inference generative models. Each module follows the **thin orchestrator pattern** where numbered scripts delegate to modular implementations.

---

## Cold Start

New to this repo as an agent? In order:

1. **What this is**: [SPEC.md](SPEC.md) + this file's Overview (25-step GNN pipeline, thin orchestrators).
2. **Exemplar spec files**: [input/gnn_files/INDEX.md](input/gnn_files/INDEX.md) — every runnable example with a selection table.
3. **Version map**: [docs/VERSION_MAP.md](docs/VERSION_MAP.md) — release history and version-sensitive surfaces.
4. **How to verify**: verification commands live in [TO-DO.md](TO-DO.md); doc audits: `uv run python docs/development/docs_audit.py --strict --check-anchors --no-write`.
---

## Module Registry

### Core Processing Modules (Steps 0-9)

- **Step 0**: **[template/](src/gnn/template/AGENTS.md)** - Pipeline template and initialization
- **Step 1**: **[setup/](src/gnn/setup/AGENTS.md)** - Environment setup and dependency management
- **Step 2**: **[tests/](tests/AGENTS.md)** - Comprehensive test suite execution
- **Step 3**: **[gnn/](src/gnn/AGENTS.md)** - GNN file discovery, parsing, and multi-format serialization
- **Step 4**: **[model_registry/](src/gnn/model_registry/AGENTS.md)** - Model versioning and registry management
- **Step 5**: **[type_checker/](src/gnn/type_checker/AGENTS.md)** - Type checking and validation
- **Step 6**: **[validation/](src/gnn/validation/AGENTS.md)** - Advanced validation and consistency checking
- **Step 7**: **[export/](src/gnn/export/AGENTS.md)** - Multi-format export generation
- **Step 8**: **[visualization/](src/gnn/visualization/AGENTS.md)** - Graph and matrix visualization
- **Step 9**: **[advanced_visualization/](src/gnn/advanced_visualization/AGENTS.md)** - Advanced visualization and interactive plots

### Simulation & Analysis Modules (Steps 10-16)

- **Step 10**: **[ontology/](src/gnn/ontology/AGENTS.md)** - Active Inference ontology processing
- **Step 11**: **[render/](src/gnn/render/AGENTS.md)** - Code generation for simulation frameworks
- **Step 12**: **[execute/](src/gnn/execute/AGENTS.md)** - Execute rendered simulation scripts
- **Step 13**: **[llm/](src/gnn/llm/AGENTS.md)** - LLM-enhanced analysis and interpretation
- **Step 14**: **[ml_integration/](src/gnn/ml_integration/AGENTS.md)** - Machine learning integration
- **Step 15**: **[audio/](src/gnn/audio/AGENTS.md)** - Audio generation and sonification
- **Step 16**: **[analysis/](src/gnn/analysis/AGENTS.md)** - Advanced statistical analysis

### Integration & Output Modules (Steps 17-24)

- **Step 17**: **[integration/](src/gnn/integration/AGENTS.md)** - System integration and coordination
- **Step 18**: **[security/](src/gnn/security/AGENTS.md)** - Security validation and access control
- **Step 19**: **[research/](src/gnn/research/AGENTS.md)** - Research tools and experimental features
- **Step 20**: **[website/](src/gnn/website/AGENTS.md)** - Static HTML website generation
- **Step 21**: **[mcp/](src/gnn/mcp/AGENTS.md)** - Model Context Protocol processing
- **Step 22**: **[gui/](src/gnn/gui/AGENTS.md)** - Interactive GUI for model construction (includes gui_1, gui_2, gui_3, oxdraw)
- **Step 23**: **[report/](src/gnn/report/AGENTS.md)** - Comprehensive analysis report generation
- **Step 24**: **[intelligent_analysis/](src/gnn/intelligent_analysis/AGENTS.md)** - AI-powered pipeline analysis and executive reports

### Infrastructure Modules

- **[utils/](src/gnn/utils/AGENTS.md)** - Shared utilities and helper functions
- **[pipeline/](src/gnn/pipeline/AGENTS.md)** - Pipeline orchestration and configuration
- **[api/](src/gnn/api/AGENTS.md)** - REST API (FastAPI)
- **[cli/](src/gnn/cli/AGENTS.md)** - `gnn` CLI entry point
- **[lsp/](src/gnn/lsp/AGENTS.md)** - Language Server Protocol support
- **[audio/sapf/](src/gnn/audio/sapf/AGENTS.md)** - SAPF audio submodule (canonical home; registered via `src/gnn/audio/mcp.py`)
- **[docs/](src/gnn/doc/AGENTS.md)** - In-repo technical documentation subtree

### v3.0.0 Long-Running Orchestration Modules (`src/gnn/pipeline/`)

Safe-by-design contracts and additive live wiring introduced in v3.0.0 ("Long-Running Orchestration"). The three core contracts are pure data layers that generate, validate, and serialize plans without mutating any live infrastructure.

- **`durable_streams.py`** - Durable observation streams: pure, replayable file/array stream manifests and traces (foundation laid before any live sensor/device stream)
- **`run_session.py`** - Resumable run-session management: durable run manifests, status inspection, and resume semantics for extended acceptance runs
- **`container_plan.py`** - Auditable container plans: validated, hardened container plans with static security review and rollback semantics
- **`session_acceptance.py`** - Live wiring that wraps the manifest-driven model-family acceptance runner in a resumable run session
- **`run_manifest.py`** - Additive, real-object emission turning a completed pipeline run's `output/` artifacts into durable, replayable v3 artifacts
- **`pipeline_container_plan.py`** - Generates an auditable, hardened container plan for running the GNN pipeline, derived from the real pipeline config

### Documentation Agents

- **[gnn/](docs/gnn/AGENTS.md)** - GNN Documentation System
- **[deployment/](docs/deployment/AGENTS.md)** - Deployment Documentation

---

## Architectural Pattern

### Thin Orchestrator Design

```mermaid
graph TB
    subgraph "Orchestrator Layer"
        Script[N_Module.py<br/>Thin Orchestrator]
    end
    
    subgraph "Module Layer"
        Init[__init__.py<br/>Public API]
        Processor[processor.py<br/>Core Logic]
        Framework[framework/<br/>Framework Code]
        MCP[mcp.py<br/>MCP Tools]
    end
    
    Script -->|Calls| Init
    Init -->|Delegates| Processor
    Processor -->|Uses| Framework
    Processor -->|Registers| MCP
    
    style Script fill:#e3f2fd
    style Init fill:#f3e5f5
    style Processor fill:#fff3e0
```

**Numbered Scripts** (`src/gnn/N_module.py`):

- Handle argument parsing via `gnn.utils.arguments.ArgumentParser`
- Setup logging via `utils.logging.logging_utils.setup_step_logging`
- Get output directories via `pipeline.config.get_output_dir_for_script`
- Call module processing functions from `module/__init__.py`
- Return standardized exit codes (0=success, 1=error, 2=warning)

**Module Implementation** (`src/gnn/<module>/`):

- Contains all domain logic in `processor.py` and subdirectories
- Provides public API via `__init__.py` exports
- Implements explicit error handling, skip statuses, and dependency diagnostics
- Registers MCP tools in `mcp.py`

### Example Structure

```
src/gnn/
├── 11_render.py              # Thin orchestrator (< 150 lines)
├── render/                   # Module implementation
│   ├── __init__.py          # Public API exports
│   ├── AGENTS.md            # This documentation
│   ├── processor.py         # Core logic
│   ├── pymdp/               # Framework-specific code
│   ├── rxinfer/
│   └── mcp.py               # MCP tool registration
```

### Function Signature Pattern

All module processing functions follow this pattern:

```python
def process_module(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs
) -> bool:
    """
    Main processing function for module.

    Parameters:
        target_dir: Directory containing input files
        output_dir: Directory for output files
        verbose: Enable verbose logging
        **kwargs: Additional module-specific options

    Returns:
        True if processing succeeded, False otherwise
    """
```

---

## Pipeline Execution Flow

```mermaid
flowchart TD
    Main[src/gnn/main.py] -->|Orchestrates| Steps[25 Pipeline Steps]
    
    Steps --> Step0[Step 0: Template]
    Step0 --> Step1[Step 1: Setup]
    Step1 --> Step2[Step 2: Tests]
    Step2 --> Step3[Step 3: GNN]
    Step3 --> Step4[Step 4: Registry]
    Step4 --> Step5[Step 5: Type Check]
    Step5 --> Step6[Step 6: Validation]
    Step6 --> Step7[Step 7: Export]
    Step7 --> Step8[Step 8: Visualization]
    Step8 --> Step9[Step 9: Advanced Viz]
    Step9 --> Step10[Step 10: Ontology]
    Step10 --> Step11[Step 11: Render]
    Step11 --> Step12[Step 12: Execute]
    Step12 --> Step13[Step 13: LLM]
    Step13 --> Step14[Step 14: ML Integration]
    Step14 --> Step15[Step 15: Audio]
    Step15 --> Step16[Step 16: Analysis]
    Step16 --> Step17[Step 17: Integration]
    Step17 --> Step18[Step 18: Security]
    Step18 --> Step19[Step 19: Research]
    Step19 --> Step20[Step 20: Website]
    Step20 --> Step21[Step 21: MCP]
    Step21 --> Step22[Step 22: GUI]
    Step22 --> Step23[Step 23: Report]
    Step23 --> Step24[Step 24: Intelligent Analysis]

    Step24 --> Output[output/]
    Output --> Summary[pipeline_execution_summary.json]
```

### Data Dependencies

Edges below describe processor-level data flow. In v4, source consumers
receive the immutable selected-model manifest, artifact consumers receive
current-run indexes, and Steps 23/24 receive the current summary snapshot.
The [migration guide](docs/development/run_ownership_migration.md) governs
identity, freshness and scheduling; directory presence cannot admit prior-run
evidence. Most steps re-parse the input GNN files directly; only the edges
shown move artifacts between steps. Solid arrows are artifact reads the
consumer performs on its primary path; dotted arrows are optional enrichments
read only when the producer's output directory exists.

```mermaid
graph TD
    IN["input/ GNN files"]

    IN -->|"glob + read"| S0["Step 0: Template"]
    IN -->|"rglob + parse; producer of parsed artifacts"| S3["Step 3: GNN Parse"]
    IN -->|"glob + read"| S4["Step 4: Model Registry"]
    IN -->|"rglob + re-parse"| S5["Step 5: Type Check"]
    IN -->|"rglob + re-parse"| S10["Step 10: Ontology"]
    IN -->|"rglob + re-parse"| S11["Step 11: Render"]
    IN -->|"rglob + re-parse"| S14["Step 14: ML Integration"]
    IN -->|"glob + scan"| S18["Step 18: Security"]
    IN -->|"rglob + re-parse"| S19["Step 19: Research"]
    IN -->|"discover + read (Step 3 JSON preferred)"| S8["Step 8: Visualization"]
    IN -->|"rglob + re-parse"| S13["Step 13: LLM"]
    IN -->|"glob + re-parse"| S15["Step 15: Audio"]
    IN -->|"rglob + re-parse"| S16["Step 16: Analysis"]
    IN -->|"discover + re-parse"| S17["Step 17: Integration"]
    IN -->|"markdown starter content"| S22["Step 22: GUI"]

    S3 -->|"gnn_processing_results.json + *_parsed.json"| S6["Step 6: Validation"]
    S3 -->|"gnn_processing_results.json + *_parsed.json"| S7["Step 7: Export"]
    S3 -->|"*_parsed.json preferred; markdown fallback"| S8
    S3 -->|"manifest + *_parsed.json"| S9["Step 9: Advanced Viz"]

    S11 -->|"rendered scripts + render manifest"| S12["Step 12: Execute"]
    S12 -->|"execution results"| S16

    S10 -.->|"ontology_results.json (when present)"| S13
    S12 -.->|"execution telemetry (when present)"| S15
    S11 -.->|"render output dir (when present)"| S17
    S12 -.->|"meta-analysis (when present)"| S17

    S12 -->|"execution summary"| S20["Step 20: Website"]
    S16 -->|"analysis JSONs"| S20

    S23["Step 23: Report"]
    S23 -.->|"census of all step output dirs; viz catalog from 8/9/11/12/20"| CENSUS["output/ (every N_*_output)"]

    SUM["output/00_pipeline_summary/pipeline_execution_summary.json"]
    SUM --> S23
    SUM --> S24["Step 24: Intelligent Analysis"]
```

| Step | Runtime input source |
|------|----------------------|
| 0, 4, 5, 10, 11, 14, 18, 19 | Re-parse `input/` GNN files from the target directory |
| 1, 2, 21 | No pipeline data (environment setup, pytest run, module discovery) |
| 3 | Parses `input/` and writes parsed artifacts to `output/3_gnn_output/` |
| 6, 7, 9 | Read Step 3 artifacts from `output/3_gnn_output/` |
| 8 | Step 3 `*_parsed.json` preferred; markdown re-parse fallback |
| 12 | Step 11 rendered scripts + render manifest from `output/11_render_output/` |
| 13 | Re-parses `input/`; optionally injects `10_ontology_output/ontology_results.json` |
| 15 | Re-parses `input/`; optionally loads Step 12 execution telemetry |
| 16 | Re-parses `input/` and reads Step 12 execution results |
| 17 | Re-parses `input/`; optional meta-analysis over Step 11/12 outputs |
| 20 | Step 16 analysis JSONs, Step 12 execution summary, Step 8/9 visualization assets |
| 22 | Input markdown starter content; navigation census of step output dirs |
| 23 | Census of all 25 step output dirs plus `pipeline_execution_summary.json` |
| 24 | `output/00_pipeline_summary/pipeline_execution_summary.json` only |

Step-orchestration prerequisites (used for `--only-steps` auto-inclusion) are
declared separately in
`src/gnn/utils/pipeline_orchestration/pipeline_step_dependencies.py` and are
deliberately broader than this artifact graph.

---

## Performance Characteristics

### Latest Status

- The pipeline contains 25 steps and follows the thin orchestrator pattern.
- Module docs and tests are maintained as part of normal development.
- Use current CI/local runs for exact pass counts and performance measurements.

### Enhanced Visual Logging Features

- **Visual Progress Indicators**: Real-time progress bars and status icons across all pipeline steps
- **Color-Coded Output**: Consistent color schemes (green=success, yellow=warning, red=error)
- **Structured Summaries**: Formatted tables showing step metrics, timing, and memory usage
- **Correlation ID Tracking**: Unique tracking IDs for debugging and monitoring pipeline execution
- **Screen Reader Support**: Accessible output with emoji-free alternatives for assistive technologies
- **Performance Monitoring**: Built-in timing and resource consumption tracking with visual displays

### Historical Validation (September 2026)

- **uv 0.12.0 compatibility**: Verified. `uv lock --check` passes and `uv sync --frozen` succeeds. `uv run --extra dev` executes all tests. The `uv` toolchain constraint `uv>=0.7.8` in Dockerfile is the minimum bootstrap floor.
- **Docs audit**: `uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write` reports no broken links, anchor gaps, or AGENTS/README coverage gaps.
- **GNN doc patterns**: `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` reports no banned GNN documentation patterns.
- **Repository terminology**: `uv run --extra dev python scripts/check_repo_terminology.py --strict` and `scripts/check_maintained_doc_terms.py --strict` report no violations.
- **Mypy**: `uv run --extra dev mypy src --config-file pyproject.toml` passes (0 errors).
- **Tests**: command of record is `uv run --extra dev python -m pytest tests/ -q --tb=no -rsx -m "not ollama"`; the release receipt for the current version (collected/passed/skipped totals on the curated exemplar corpus) is recorded in `CHANGELOG.md` §3.3.0 (2026-09-07); re-run the command for live totals rather than trusting numbers in prose. Julia RxInfer execution uses the committed `Project.toml` under `src/gnn/execute/rxinfer/` (RxInfer 5.5.0 pinned); ActiveInference.jl uses the committed minimal env under `src/gnn/execute/activeinference_jl/`. With a local Ollama daemon and `smollm2:135m-instruct-q4_K_S` pulled, the two Ollama files are re-enabled.
- **LLM Default Model**: `smollm2:135m-instruct-q4_K_S` via Ollama (`llm.defaults.DEFAULT_OLLAMA_MODEL`; override with `OLLAMA_MODEL` / `input/config.yaml`).
- **Renderer inventory**: derive maintained paths and model-family support from `src/gnn/render/framework_registry.py`. Implemented bnlearn and ngc-learn execution is declared in the executor registry. Experimental cpomdp and THRML require explicit framework selection; their adapter-specific evidence is separate from the historical public GridWorld output.
- **Default dev suite**: FastAPI, websocket bridge, and LSP tests run under the `dev` extra; browser, public-network, live GUI, audio-DSP, and Ollama integrations remain explicit opt-in surfaces rather than hidden default-suite skips.
- **Visual Accessibility**: All pipeline steps now include enhanced visual indicators and progress tracking.

### v3.2.0 "Exemplar Gold Standard" (2026-09-02)

- **Two model kinds**: `render.pomdp_contract.detect_model_kind` classifies each spec as discrete (categorical `A/B/C/D[/E]`) or continuous; the `input/gnn_files/continuous/` exemplars declare only the linear-Gaussian state-space block (`F/H/Q/R`, `prior_mean/prior_cov`, optional closed-loop `goal_mean/control_gain`) and are rendered verbatim, never canonicalised to A/B/C/D.
- **`unsupported` render status**: frameworks whose `framework_registry.py` entry has `supports_continuous: False` (PyMDP, ActiveInference.jl, DisCoPy, bnlearn) return `status: unsupported` for continuous models; these are counted separately under `unsupported_framework_renderings` in `render_processing_summary.json`, excluded from success rates, and never executed by Step 12.
- **Native continuous backends**: JAX, NumPyro (+NUTS), PyTorch and Stan share `render/continuous_script.py` (online Kalman filter, Joseph-form update, closed-loop control); RxInfer.jl uses its native LGSSM strategy.
- **Stan is executable**: `render/stan/stan_renderer.py` emits an HMM (forward-algorithm marginalisation, Dirichlet priors centred on `A`) for discrete models and the Kalman marginal likelihood for continuous ones, as `<stem>_stan.stan` plus a `<stem>_stan.py` cmdstanpy driver; `src/gnn/execute/stan/` runs it and `gnn.utils.runtime_safety.framework_availability` reports `skipped` when `cmdstanpy`/CmdStan is absent (`uv sync --extra stan`).
- **Historical Step 12 summary merge**: v3.2 combined summaries across input folders. v4 aggregates the frozen current-run selection once and excludes inherited execution artifacts from current evidence.
- **Julia pre-exec gate**: a `julia` launcher without a working toolchain no longer blocks scripts; the probe degrades to the advisory regex sweep unless the parser itself reports a failure.
- Live counts come from `output/11_render_output/render_processing_summary.json` and `output/12_execute_output/summaries/execution_summary.json`; see `CHANGELOG.md` §3.2.0 for the release receipt.

### v3.3.0 "One Corpus" (2026-09-06)

- The `src.*` import surface is gone: `gnn.*` is the single canonical import
  surface (from an installed wheel, or with `PYTHONPATH=src` from the repo), and
  `src/` holds only the `gnn` package.
- Every model file lives inside `input/gnn_files`; the former top-level
  `multi_agent_models/` and `recursive_models/` directories are folded into
  `multiagent/` and `recursive/` respectively.
- Headless extraction (`gnn extract FILE` / `python -m gnn.extract`), canonical
  B orientation with `canonicalize_pomdp()`, factor/dimension provenance, and
  the `torch` optional extra are documented in `CHANGELOG.md` §3.3.0.

### Current publication: GNN 4.0.1 (2026-10-07)

The [patch publication receipt](docs/development/gnn_4_0_1_post_publication.json)
records source/tag identities, exact GNN/FEP/GEO gates, issue acceptance and
provider alert closure. Preserve historical receipts and their source epochs.
The package is 4.0.1; future capability limits remain in [TO-DO.md](TO-DO.md).

### v4.0.0 "Current-run Reliability" (2026-10-01 implementation)

- Frozen selection, path-derived identities, source hashes, resolved configuration,
  current-run artifact indexes, output leases, and one monotonic deadline bind
  each invocation. Required unfinished work prevents success. Read the
  [migration guide](docs/development/run_ownership_migration.md) before changing
  these contracts or consuming earlier artifacts as evidence for a new run.
- [THRML](docs/gnn/implementations/thrml.md) uses the ordinary released
  `thrml==0.1.4` wheel for admitted finite categorical Gibbs smoothing under
  fixed actions. Pure validation/admission, generated native sampling,
  supervised execution, and sample-bound analysis are separate modules.
  Configure `render.backend_options.thrml`; direct CLI/API/MCP options share the
  renderer's validated option set. Structural zeros, coupled agents, continuous models,
  action search, convergence, and hardware operation are outside this contract.
- [cpomdp](docs/gnn/implementations/cpomdp.md) remains an explicitly selected
  experimental released-wheel backend. Admission estimates, numerical witnesses,
  execution receipts, and formal custody retain separate claims.
- Step 16 admits numerical comparisons only for compatible model IDs, source
  hashes, inference semantics, inputs and declared precision. Native Gaussian
  uncertainty comes from covariance; unbound or incompatible comparisons retain
  refusal reasons through JSON, Markdown and plotting.
- The [scope ledger](SCOPE-2026-10-01.md) records acceptance and deferred checks.
  Publication requires final repository gates, manuscript rendering, paired
  custody, hosted results, and remote commit parity; implementation prose is not
  a substitute for those receipts.

---

## Module Status Matrix

Module-level status is maintained in each module's own `AGENTS.md` and test files. As of v3.0.0, the `src/gnn/pipeline/` orchestration contracts (`durable_streams`, `run_session`, `container_plan`) and their live wiring (`session_acceptance`, `run_manifest`, `pipeline_container_plan`) are validated by the strict `scripts/run_v3_orchestration_acceptance.py` gate.

---

## Quick Start

### Using `just` (Recommended)

```bash
brew install just              # macOS — or: cargo install just
just                           # See available recipes
just test                      # Fast test suite
just lint                      # Ruff lint check
just pipeline                  # Full 25-step pipeline
just pipeline-steps "3,5,11,12"  # Specific steps
just render-health             # Check renderer availability
just test-mod render           # Test a specific module
```

### Direct Commands

#### Run Full Pipeline

```bash
python src/gnn/main.py --target-dir input/gnn_files --verbose
```

#### Run Specific Steps

```bash
python src/gnn/main.py --only-steps "3,5,7,8,11,12" --verbose
```

#### Run Individual Step

```bash
python src/gnn/3_gnn.py --target-dir input/gnn_files --output-dir output --verbose
```

---

## Development Guidelines

### Adding New Modules

1. Create module directory: `src/gnn/new_module/`
2. Implement `__init__.py` with public API
3. Create `AGENTS.md` documentation
4. Add numbered script: `N_new_module.py`
5. Implement tests in `tests/`
6. Add MCP tools in `mcp.py` (if applicable)

### Code Standards

- Follow thin orchestrator pattern
- Use type hints for all public functions
- Document all classes and methods
- Maintain >80% test coverage
- Include explicit error handling, status reporting, and dependency diagnostics

---

## Testing

### Run All Tests

```bash
python src/gnn/2_tests.py --comprehensive
```

### Run Module-Specific Tests

```bash
uv run --extra dev python -m pytest tests/test_[module]*.py -v
```

### Check Coverage

```bash
pytest --cov=src --cov-report=term-missing
```

---

## Cross-repo custody (fep_lean, GEO-INFER)

This repository shares custody pairs with two sibling repositories: fep_lean (bridge contract) and GEO-INFER (interchange contracts); each side pins the other via committed pin files plus paired CI. The one-line rule: **any edit to `src/gnn/**/*.py` (or `pyproject.toml`, `uv.lock`, `src/gnn/main.py`, `docs/gnn/gnn_syntax.md`, `src/gnn/pipeline/step_registry.py`) re-drifts the fep_lean custody pair** and must be closed out by the paired re-pin ritual. Canonical ordering and workflow details: [docs/development/fep_lean_paired_revision.md](docs/development/fep_lean_paired_revision.md).

## Manuscript custody (SC-22): count-changing PRs run the full ritual

`scripts/check_hydrated_prose.py` (the `Hydrated prose matches token map` step of `local-gates.yml`'s `repo-gates` job, `just hydrated-prose`) re-substitutes `manuscript/*.md` from the committed `output/data/manuscript_variables.json` and fails when the committed `output/manuscript/` differs. **Consequence: any PR that regenerates the token map (every count-changing `src/` or `tests/` PR) must run the full SC-22 ritual before it can go green.** Regenerating the map and re-recording the custody manifest is no longer enough. The ritual: `python -m scripts.manuscript_build_figures` → the template's `stage_03_render` (fresh `docxology/template` clone with this checkout symlinked at `projects/active/GeneralizedNotationNotation`) → `scripts/z_verify_fresh_render.py` → `scripts/z_record_manuscript_render_manifest.py` → commit `output/`. The canonical statement is the `scripts/z_generate_manuscript_variables.py` docstring.

If manuscript custody is red on main, repair main first. PR gates use the
merge-base with the target SHA; only unchanged diagnostic and exact-evidence
fingerprints may warn with the full base SHA. New or changed drift fails.
Main and scheduled audits remain strict.

## Agent Capabilities

Each module provides specialized agent capabilities for different aspects of Active Inference model processing:

### 🎯 **Template Agent** - Intelligent Pipeline Initialization

- Dynamic configuration generation
- Context-aware template selection
- Automated dependency resolution
- Performance-optimized execution planning

### 🔧 **Setup Agent** - Environment Management

- Intelligent dependency resolution
- Virtual environment optimization
- Platform-specific configuration
- Automated security scanning

### 🧪 **Test Agent** - Quality Assurance

- Comprehensive test orchestration
- Performance benchmarking
- Coverage analysis and reporting
- Regression detection and alerting

### 📄 **GNN Agent** - Model Processing

- Multi-format file discovery
- Intelligent parsing and validation
- Semantic analysis and inference
- Cross-format data transformation

### 📋 **Registry Agent** - Model Management

- Version control and tracking
- Metadata extraction and indexing
- Model comparison and analysis
- Provenance and lineage tracking

### ✅ **Type Checker Agent** - Validation

- Static type analysis
- Resource estimation and optimization
- Constraint verification
- Performance prediction modeling

### 🔍 **Validation Agent** - Consistency Checking

- Cross-reference validation
- Logical consistency verification
- Mathematical constraint checking
- Domain-specific rule enforcement

### 📤 **Export Agent** - Format Translation

- Multi-format data export
- Schema transformation
- Metadata preservation
- Format-specific optimization

### 🎨 **Visualization Agent** - Graph Generation

- Network topology visualization
- Matrix heatmap generation
- Interactive diagram creation
- Performance metric plotting

### 🔬 **Advanced Visualization Agent** - Enhanced Graphics

- 3D visualization generation
- Interactive dashboard creation
- Real-time data streaming
- Custom visualization frameworks

### 🧠 **Ontology Agent** - Knowledge Processing

- Active Inference term mapping
- Semantic relationship discovery
- Knowledge graph construction
- Domain-specific reasoning

### ⚙️ **Render Agent** - Code Generation

- Multi-framework code generation
- Language-specific optimization
- Framework-specific templates
- Performance-tuned implementations

### 🚀 **Execute Agent** - Simulation Runner

- **ActiveInferenceAgent**: Primary full-fidelity execution engine
- Multi-environment execution follows the live executor registry, including
  PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan,
  bnlearn, ngc-learn, and explicitly selected experimental THRML/cpomdp.
- Resource monitoring and optimization
- Explicit failure, skip, and retry reporting
- Cross-platform compatibility

### 🤖 **LLM Agent** - AI Enhancement

- Natural language analysis
- Model interpretation and explanation
- Automated documentation generation
- Multi-modal reasoning

### 🔗 **ML Integration Agent** - Machine Learning

- Model training and evaluation
- Hyperparameter optimization
- Performance comparison
- Integration with ML frameworks

### 🎵 **Audio Agent** - Sonification

- Multi-backend audio generation
- Real-time audio processing
- Audio feature extraction
- Sonification of model dynamics

### 📊 **Analysis Agent** - Statistical Processing

- Advanced statistical analysis
- Performance metric computation
- Trend analysis and forecasting
- Anomaly detection

### 🔗 **Integration Agent** - System Coordination

- Cross-module data flow
- Pipeline orchestration
- Resource allocation
- Inter-module communication

### 🔒 **Security Agent** - Protection

- Input validation and sanitization
- Access control implementation
- Threat detection and mitigation
- Compliance verification

### 🔬 **Research Agent** - Experimental Tools

- Research workflow management
- Experimental design assistance
- Literature review automation
- Collaboration tools

### 🌐 **Website Agent** - Documentation

- Static site generation
- Documentation compilation
- Cross-reference linking
- Search and navigation

### 🔗 **MCP Agent** - Protocol Integration

- Tool registration and discovery
- Protocol compliance
- Cross-system communication
- Standard interface implementation

### 🖼️ **GUI Agent** - Interactive Interfaces

- Multi-modal GUI generation
- Real-time interaction
- User experience optimization
- Accessibility compliance

### 📋 **Report Agent** - Documentation

- Comprehensive report generation
- Multi-format output
- Executive summary creation
- Performance visualization

### ⏳ **Long-Running Orchestration (v3.0.0)** - Durable, Safe-by-Design Runs

Released in v3.0.0 ("Long-Running Orchestration"), this capability adds three safe-by-design `src/gnn/pipeline/` contracts plus additive live wiring, with no live infrastructure mutation:

- **Durable observation streams** (`durable_streams.py`): replayable file/array stream manifests and traces for long observation runs
- **Resumable run sessions** (`run_session.py`, wired live via `session_acceptance.py`): durable, resumable manifests for extended model-family acceptance runs
- **Auditable container plans** (`container_plan.py`, `pipeline_container_plan.py`): validated, hardened container plans with static security review and rollback semantics, derived from the real pipeline config
- **Durable run artifacts** (`run_manifest.py`): turns a completed run's `output/` into replayable v3 artifacts
- **3 new MCP tools** exposing these contracts to agents
- **Strict acceptance gate**: `scripts/run_v3_orchestration_acceptance.py` exercises all three contracts end to end with positive and negative checks and fails closed

```bash
uv run --extra dev python scripts/run_v3_orchestration_acceptance.py
```

---

## Learned User Preferences

- When implementing an attached plan, do not edit the plan file itself; use the existing to-do items and complete them without recreating them.
- Documentation fixes should verify file paths, commands, and signposts against the current repository rather than preserving stale examples.
- Prefer measured verification outputs over estimates when reporting test, pipeline, or scaling behavior.

## Learned Workspace Facts

- Pipeline `--target-dir` arguments must point to directories; single `.md` file paths are not discovered by the current `Path.rglob`-based flow.
- The PyMDP scaling study expands dense `B` tensors as O(n^3) text, so large `N` values require explicit file-size and disk-space guardrails.
- Step 12 execution runs rendered `.py` and `.jl` scripts via the execute processor; auxiliary per-framework helper scripts are not the primary pipeline path.

---

## References

- **Main Documentation**: [README.md](README.md)
- **Architecture Guide**: [ARCHITECTURE.md](ARCHITECTURE.md)
- **Pipeline Rules**: [.agent_rules](.agent_rules)

---

**Last Updated**: 2026-10-02
**Pipeline Version**: [pyproject.toml](pyproject.toml) (canonical)
**Total Steps**: 25 (0-24)
**Status**: Maintained
