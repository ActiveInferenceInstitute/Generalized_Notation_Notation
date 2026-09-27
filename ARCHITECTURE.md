# GNN Architecture Guide

This guide details the architecture of the Generalized Notation Notation (GNN) system. It complements `DOCS.md` and `docs/pipeline/README.md` with an implementation-oriented perspective for developers.

**Last Updated**: 2026-09-21
**Version**: [pyproject.toml](pyproject.toml) (canonical · 3.6.0)
**Status**: Maintained
**Pipeline Steps**: 25 (0-24)

## Principles

### Core Architectural Principles

- **Thin Orchestrators**: Numbered scripts delegate to modules, maintaining clear separation of concerns
- **Explicit Dependencies**: All module dependencies are explicitly declared and validated
- **Deterministic Outputs**: Pipeline runs produce identical results given identical inputs
- **Reproducible Runs**: Complete audit trail and environment capture for scientific reproducibility
- **Standardized Interfaces**: Consistent exit codes, logging, and configuration patterns across all modules

### Quality Assurance Principles

- **Real Implementation Testing**: All tests use real code paths and actual data dependencies
- **Comprehensive Coverage**: Use current CI/local test output for exact coverage and pass counts
- **Real Data Processing**: Tests and examples use explicit, runnable data
- **Performance Standards**: Publish performance and memory figures only from current measured runs
- **Error Rate Targets**: Publish failure-rate claims only from current measured runs

### Agent Architecture Principles

- **Specialized Capabilities**: Each module provides distinct, well-defined agent capabilities
- **Stateless Design**: Prefer stateless agents for better testability and reliability
- **Resource Awareness**: Proper resource management with cleanup and monitoring
- **Configuration Flexibility**: Environment-driven configuration with validation
- **Integration Ready**: All modules integrate seamlessly with the 25-step pipeline

## System Overview

The GNN system implements a comprehensive 25-step pipeline that transforms GNN model specifications into executable simulations, visualizations, and analysis. The architecture follows the thin orchestrator pattern with complete separation between pipeline orchestration and domain-specific implementations.

```mermaid
graph TB
  A["User/Researcher"] --> B["src/gnn/main.py<br/>Pipeline Orchestrator"]
  B --> C["25 Numbered Scripts<br/>(0_template.py → 24_intelligent_analysis.py)"]
  C --> D["Module set<br/>(see src/gnn/AGENTS.md)"]
  D --> E["Structured Outputs<br/>(output/step_N_output/)"]

  B --> F["Infrastructure Layer<br/>(utils/, pipeline/)"]
  F --> B

  D --> G["External Integrations<br/>(PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn)"]
  D --> H["AI Services<br/>(Ollama, OpenAI, OpenRouter, Perplexity)"]
  D --> I["Scientific Frameworks<br/>(JAX, DisCoPy, NetworkX)"]

  %% styling intentionally omitted (theme-controlled)
```

## Execution Flow (High-Level)

The GNN pipeline implements a sophisticated execution model with comprehensive monitoring, error recovery, and performance optimization.

```mermaid
sequenceDiagram
  participant U as User/Researcher
  participant M as src/gnn/main.py
  participant S as Step N Script
  participant Mod as Agent Module
  participant Out as output/*
  participant Ext as External Services

  U->>M: CLI arguments & config
  M->>S: Invoke with parsed args + context
  S->>Mod: Call domain-specific APIs
  Mod->>Ext: Optional external integrations
  Ext-->>Mod: Results or service responses
  Mod-->>S: Structured results + diagnostics
  S-->>M: Exit codes + performance metrics
  M->>Out: Comprehensive output artifacts
  M->>M: Update pipeline summary & health report
```

## Thin Orchestrator Pattern

- Orchestrators handle: argument parsing, logging setup, output dir, calling module APIs, summarizing results
- Modules implement: domain logic, IO, validations, transformations, rendering, execution

```mermaid
graph LR
  O["Orchestrator N_step.py"] -->|calls| API[("Module Public API")]
  API --> Impl["Internal Logic"]
  API --> IO["IO / Files"]
  Impl --> Artifacts["Artifacts in output/"]
```

## Module Dependencies

The GNN pipeline implements a sophisticated dependency graph that ensures proper execution order and data flow between modules.

```mermaid
graph TD
  GNN[gnn] --> TYPE[type_checker]
  GNN --> REG[model_registry]
  GNN --> VAL[validation]
  GNN --> EXP[export]
  GNN --> ONT[ontology]
  GNN --> RENDER[render]
  GNN --> LLM[llm]
  GNN --> AUDIO[audio]
  GNN --> RES[research]
  GNN --> GUI[gui]

  TYPE --> VAL
  VAL --> EXP
  EXP --> VIS[visualization]
  EXP --> AV[advanced_visualization]

  RENDER --> EXEC[execute]
  RENDER --> SEC[security]

  EXEC --> ANA[analysis]
  VIS --> ANA
  LLM --> ANA
  ML[ml_integration] --> ANA

  ANA --> REP[report]
  REP --> IA[intelligent_analysis]

  GNN --> INT[integration]
  GNN --> WEB[website]
  GNN --> MCP[mcp]

  %% styling intentionally omitted (theme-controlled)
```

## Long-Running Orchestration (v3.0.0)

Version 3.0.0 ("Long-Running Orchestration") adds a safe-by-design surface under `src/gnn/pipeline/` for durable, resumable, and auditable pipeline runs. The defining property is that no module mutates live infrastructure: every contract generates, validates, replays, or plans **data only**, so the entire surface is inspectable before anything is acted upon. See [docs/pipeline/v3_orchestration.md](docs/pipeline/v3_orchestration.md) for the full reference.

### Three Safe-by-Design Contracts

- **Durable observation streams** (`src/gnn/pipeline/durable_streams.py`): append-only, replayable records of pipeline observations. Streams are written and read back as synthetic data, never as live side effects.
- **Resumable run sessions** (`src/gnn/pipeline/run_session.py`): session state that captures progress so a run can be resumed deterministically. Sessions describe what would resume; they do not execute steps themselves.
- **Auditable container plans** (`src/gnn/pipeline/container_plan.py`): declarative container execution plans emitted as inspectable data, allowing the intended environment to be reviewed without launching any container.

### Data-Only Inspectable Surface

The three contracts compose into a single inspectable surface. Each produces a serializable artifact — a stream, a session record, or a plan — that can be validated and replayed independently. Because nothing in this layer performs live mutation, an operator (or an automated gate) can examine the complete intended behavior of a long-running run before it touches any real resource.

### Additive Live Wiring

The contracts are wired into the pipeline additively, leaving existing behavior unchanged:

- `src/gnn/pipeline/session_acceptance.py` — validates session state against acceptance criteria.
- `src/gnn/pipeline/run_manifest.py` — assembles the run manifest from session and stream data.
- `src/gnn/pipeline/pipeline_container_plan.py` — derives container plans from pipeline configuration.

A strict acceptance gate, `scripts/run_v3_orchestration_acceptance.py`, exercises these contracts end-to-end, and three new MCP tools expose the orchestration surface to agent clients.

## Exemplar Gold Standard (v3.2.0)

Version 3.2.0 makes every exemplar under `input/gnn_files/` render *and* execute on every framework that can represent it, and flag the rest explicitly:

- **Model kinds**: `src/gnn/render/pomdp_contract.py` (`detect_model_kind`) classifies every spec: a discrete kind (flat, factored, hierarchical, multi-agent, learning, or the structural no-parameterization wrapper) over the categorical `A/B/C/D[/E]` contract, or the continuous linear-Gaussian kind (`F/H/Q/R`, `prior_mean/prior_cov`, optional closed-loop `goal_mean/control_gain`). Continuous blocks pass through the render processor verbatim.
- **Framework capabilities**: `src/gnn/render/framework_registry.py` is the single declaration of the ten frameworks and carries `supports_continuous` per entry. Frameworks without continuous support (PyMDP, ActiveInference.jl, DisCoPy, bnlearn) return the `unsupported` render status for continuous models; it is counted separately in `render_processing_summary.json` and never reaches Step 12.
- **Shared LGSSM generator**: `src/gnn/render/continuous_script.py` produces the online Kalman filter (Joseph-form update, closed-loop control) used by the JAX, NumPyro, PyTorch and Stan renderers; RxInfer.jl keeps its native continuous strategy.
- **Stan execution**: `src/gnn/render/stan/stan_renderer.py` emits runnable HMM and LGSSM programs plus a cmdstanpy driver, and `src/gnn/execute/stan/` runs them (skipped, not failed, without `cmdstanpy`/CmdStan).
- **Step 12 summary merge**: `src/gnn/execute/processor.py` merges the prior `execution_summary.json` so one durable summary covers every input folder, mirroring Step 11.

See `CHANGELOG.md` §3.2.0 and the README section "Model Kinds and Framework Support".

## One Corpus (v3.3.0)

Version 3.3.0 folds every model file under `input/` into the `input/gnn_files/`
corpus tree, makes `gnn.*` the single canonical import surface (the `src.*`
surface was removed by the package rename), and adds headless extraction
(`gnn extract FILE` / `python -m gnn.extract`), canonical
`(next_state, previous_state, action)` B-orientation enforcement with
`canonicalize_pomdp()`, factor counts and dimension provenance in the extractor
payload, durable `gnn-run-v2` run identity with verified `gnn reproduce`, and a
dedicated `torch` optional extra for the Step 11/12 PyTorch backend. The
manuscript remediation gates are reconciled onto `src/gnn/`. See
`CHANGELOG.md` §3.3.0.

## Current Implementation Status

### Components

**Core Infrastructure:**

- `src/gnn/main.py` - Main pipeline orchestrator with comprehensive monitoring
- `src/gnn/utils/` - Complete utility library with logging, validation, and monitoring
- `src/gnn/pipeline/` - Full pipeline configuration and management system
- `tests/` - Comprehensive test suite with real data validation

**Agent Modules:**

- All 25 pipeline steps (0-24) implemented with thin orchestrator pattern
- Agent/module coverage is tracked in `src/gnn/AGENTS.md`
- Complete MCP integration across all applicable modules
- Repository coverage is enforced by the live `pyproject.toml` policy
  (`fail_under = 50`); use the generated coverage report for per-module detail

**Documentation:**

- `AGENTS.md` - Master agent scaffolding documentation
- `AGENTS_TEMPLATE.md` - Template for new modules
- `.agent_rules` - Development guidelines
- `.env` (optional local file) - local environment values when needed
- `.gitignore` - Comprehensive ignore patterns for scientific computing

### Latest Status

- The architecture uses thin orchestrators over modular implementations.
- Pipeline, docs, and tests are maintained together.
- Use current runs and reports for exact performance and pass metrics.

## Logging Architecture

```mermaid
flowchart TD
  A["setup_step_logging"] --> B["Module Logger"]
  B --> C["Per-step Logs"]
  C --> D["Aggregated Pipeline Logs"]
```

## Configuration Flow

```mermaid
flowchart LR
  CLI["CLI Args"] --> Merge
  CFG["config.yaml"] --> Merge
  ENV["Env Vars"] --> Merge
  Merge["Configuration Resolver"] --> Eff["Effective Config"]
  Eff --> Steps["Steps 0..24"]
```

## Output Management

- Each step writes to `output/<step_subdir>/`
- `get_output_dir_for_script()` ensures consistent paths
- Site and reports summarize artifacts across steps

## Error Handling

- Exit codes: 0=success, 1=error, 2=success with warnings/skipped
- Continuation policy controlled via config (fail-fast vs continue)
- Rich diagnostics persisted alongside artifacts

```mermaid
flowchart LR
  S["Step Start"] --> V{Valid?}
  V -- no --> E["Log + Diagnostics"]
  E --> P{Fail-fast?}
  P -- yes --> STOP["Abort"]
  P -- no --> CONT["Continue"]
  V -- yes --> RUN["Run"]
  RUN --> X{Exit Code}
  X -->|0| OK["Success"]
  X -->|2| WARN["Success+Warnings"]
  X -->|1| E
```

## Extension Pattern

Adding new pipeline steps and modules follows a well-established pattern that ensures consistency and maintainability:

### 1. **Plan the Extension**

- Define the module's purpose and integration points
- Identify dependencies on existing modules
- Determine resource requirements and performance characteristics
- Plan comprehensive test coverage and documentation

### 2. **Create Thin Orchestrator**

- Implement `src/N_newstep.py` following the thin orchestrator pattern
- Handle argument parsing, logging setup, and output directory management
- Delegate all domain logic to the module implementation
- Return standardized exit codes (0=success, 1=error, 2=success with warnings/skipped)

### 3. **Implement Agent Module**

- Create `src/newstep/` directory with complete module structure
- Implement `__init__.py` with public API exports
- Create core logic in `processor.py` or appropriately named files
- Add `mcp.py` for Model Context Protocol integration (if applicable)
- Include comprehensive error handling and resource management

### 4. **Add Comprehensive Testing**

- Create integration tests in `tests/newstep/test_newstep_integration.py`
- Implement unit tests for all public functions
- Add performance tests with timing and memory validation
- Include error scenario testing with real failure modes
- Meet or raise the coverage floor declared in `pyproject.toml`, with tests
  focused on the new behavior and failure modes

### 5. **Document Completely**

- Create comprehensive `AGENTS.md` using the enhanced template
- Document all public APIs with examples and error conditions
- Include troubleshooting guide and performance characteristics
- Add usage examples for common scenarios
- Update pipeline documentation and cross-references

### 6. **Validate and Review**

- Test the complete integration with existing pipeline
- Validate performance against established standards
- Ensure compliance with all coding standards and patterns
- Review security implications and access controls
- Update configuration files and environment templates as needed

## Agent Architecture Deep Dive

The GNN system implements a sophisticated multi-agent architecture where each module provides specialized capabilities:

### 🤖 **Agent Types and Capabilities**

**Processing Agents** (Steps 3-9):

- **GNN Agent**: Multi-format parsing and semantic analysis
- **Type Checker Agent**: Static validation and resource estimation
- **Validation Agent**: Consistency checking and constraint verification
- **Export Agent**: Multi-format data transformation and serialization
- **Visualization Agent**: Graph generation and matrix visualization
- **Advanced Visualization Agent**: Interactive plots and 3D graphics

**Simulation Agents** (Steps 10-16):

- **Ontology Agent**: Active Inference knowledge processing and mapping
- **Render Agent**: Multi-framework code generation and optimization
- **Execute Agent**: Cross-platform simulation execution and monitoring
- **LLM Agent**: AI-enhanced analysis and natural language processing
- **ML Integration Agent**: Machine learning model training and evaluation
- **Audio Agent**: Multi-backend audio generation and sonification
- **Analysis Agent**: Advanced statistical processing and performance analysis

**Integration Agents** (Steps 17-24):

- **Integration Agent**: Cross-module coordination and data flow management
- **Security Agent**: Input validation, access control, and threat detection
- **Research Agent**: Experimental tools and workflow management
- **Website Agent**: Static site generation and documentation compilation
- **MCP Agent**: Protocol compliance and tool registration
- **GUI Agent**: Interactive interface generation and user experience
- **Report Agent**: Comprehensive analysis reporting and visualization
- **Intelligent Analysis Agent**: Cross-step meta-analysis over the pipeline's own outputs (Step 24)

### 🔧 **Agent Coordination Mechanisms**

**Dependency Resolution**: Automatic dependency detection and inclusion
**Resource Management**: Coordinated resource allocation and cleanup
**Error Propagation**: Structured error handling with graceful degradation
**Performance Monitoring**: Cross-agent performance tracking and optimization
**Configuration Management**: Centralized configuration with module-specific overrides

### 📊 **Agent Performance Framework**

Each agent implements comprehensive performance monitoring:

- **Resource Tracking**: Memory, CPU, and disk usage monitoring
- **Execution Timing**: Detailed timing for all major operations
- **Error Metrics**: Success rates, failure modes, and recovery statistics
- **Integration Health**: Dependency health and communication status
- **Scalability Metrics**: Performance characteristics across different input sizes

## References

- **Main Documentation**: [README.md](README.md) — Project overview and quick start
- **Pipeline Documentation**: [docs/PIPELINE_SCRIPTS.md](docs/PIPELINE_SCRIPTS.md) — Detailed step-by-step descriptions
- **Development Rules**: [.agent_rules](.agent_rules) — Canonical rules for scripts and modules
- **Agent Registry**: [AGENTS.md](AGENTS.md) — Master agent scaffolding and module registry
- **Template Guide**: [AGENTS_TEMPLATE.md](AGENTS_TEMPLATE.md) — Template for new modules
- **Doc link checks**: [DOCS.md](DOCS.md) — “Documentation maintenance” (`docs/development/docs_audit.py`)

---

**Architecture Version**: [pyproject.toml](pyproject.toml) (canonical · 3.6.0)
**Last Updated**: 2026-09-21
**Status**: Maintained
**Compliance**: Thin orchestrator pattern
**Latest Validation**: See current test and pipeline runs
