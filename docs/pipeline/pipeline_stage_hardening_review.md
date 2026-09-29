# Pipeline Stage Hardening Review

**Last reviewed**: 2026-05-22

This review is the maintained goal record for repo-wide GNN pipeline hardening. It complements the exhaustive [source step index](../../src/gnn/STEP_INDEX.md) by focusing on the operating contract each stage must keep: configurable inputs, documented outputs, structured logging, explicit validation and failure status, focused tests, and downstream handoff.

## End-to-End Proof Path

The strict generative-model proof path is the maintained 3x3 GridWorld POMDP fixture:

```bash
uv run python src/gnn/main.py --only-steps "3,5,8,11,12,16" --target-dir input/gnn_files/pomdp_gridworld --frameworks "pymdp,rxinfer,activeinference_jl" --verbose
```

Acceptance for this command:

- Step 3 parses one GridWorld GNN source into canonical model data.
- Step 5 validates dimensions and type/resource estimates.
- Step 8 produces graph and matrix visualizations.
- Step 11 renders PyMDP, RxInfer.jl, and ActiveInference.jl from the same GNN matrices.
- Step 12 executes exactly the current rendered scripts requested by `--frameworks` and collects one primary `simulation_results.json` per framework.
- Step 16 reads only current execution outputs and produces statistics, PNG plots, per-framework GIFs, a cross-framework GridWorld GIF, dashboards, and `cross_framework/gridworld_analysis_manifest.json`.
- The strict execution schemas are `pymdp_simulation_v1`, `rxinfer_simulation_v1`, and `activeinference_jl_simulation_v1`; the accepted manifest schema is `gridworld_analysis_manifest_v1`.
- The manifest's `frameworks` list must contain the three strict targets as a subset: ungated runners executed by `--frameworks all` (`numpyro_simulation_v1`, `pytorch_simulation_v1`) may appear alongside them. All five recognized runners emit the spec's `matrix_provenance` (the key is not enforced; a payload without it records `{}`), and `matrix_provenance_equal` must be true across every listed payload (`scripts/check_pomdp_gridworld_outputs.py`).

Generated outputs remain under ignored output trees and are regenerated as evidence, not maintained as source.

## Stage Review Matrix

| Step | Contract | Config, Logging, Failure Semantics | Focused Verification |
| ---: | --- | --- | --- |
| 0 | Template initializes pipeline scaffolding and metadata. | Controlled by global step selection; logs setup context; returns standard success/error/warning status. | Step script and template module tests; `src/gnn/STEP_INDEX.md` coverage. |
| 1 | Setup validates `uv`, Python, dependency groups, and environment readiness. | `--dev`, optional groups, and config defaults drive installs; dependency absence is explicit. | Setup tests, `uv lock --check`, environment tests. |
| 2 | Tests step runs configured pytest suites and writes test summaries. | CLI flags choose fast/comprehensive behavior; failures are reported in structured output. | `just test`, `just test-full`, collect-only inventory. |
| 3 | GNN discovers maintained model files, parses them, and emits canonical serialized model artifacts. | `--target-dir`, recursion, and serializer preset are configurable; per-file parse issues are logged. | GNN parser/processor tests and GridWorld extraction tests. |
| 4 | Model registry records model metadata and lookup state. | Re-globs and re-reads target-dir GNN files; registry gaps are reported without mutating unrelated models. | Model registry round-trip and overall tests. |
| 5 | Type checker validates dimensions, resource estimates, and structural consistency. | Reads target/model artifacts; validation findings are reported as structured issues. | Type checker tests and GridWorld dimension assertions. |
| 6 | Validation performs semantic, consistency, and quality checks. | Reads the Step 3 manifest and parsed models from `output/3_gnn_output/`; the source directory is recorded as informational only. | Validation module tests and pipeline handoff tests. |
| 7 | Export writes supported machine formats from parsed GNN models. | Export formats and output directory are configurable; individual format failures are isolated. | Export round-trip and pipeline export tests. |
| 8 | Visualization creates graph, matrix, ontology, and tensor views. | Uses parsed model data when available; visual backend issues produce explicit reports. | Visualization matrix/artifact tests and GridWorld pipeline checks. |
| 9 | Advanced visualization creates richer interactive/dashboard outputs. | Visualization types and layout options are configurable; unsupported tools are surfaced. | Advanced visualization shared and overall tests. |
| 10 | Ontology maps model terms to Active Inference ontology records. | Missing terms are reported as validation details, not hidden. | Ontology annotation and overall tests. |
| 11 | Render turns validated models into framework-specific scripts. | `--frameworks` selects targets; strict validation fails requested unsupported shapes clearly. | Render target tests and GridWorld PyMDP/RxInfer/ActiveInference.jl render checks. |
| 12 | Execute runs current rendered scripts and collects primary simulation logs. | Framework selection, timeout, workers, and benchmark knobs are configurable; requested framework failures are explicit. | Execute contract tests and GridWorld cross-framework execution checks. |
| 13 | LLM processing produces optional model interpretation. | Provider/model/timeouts are configurable; unavailable providers are reported. | LLM functional tests; Ollama integration stays opt-in when the daemon/model is absent. |
| 14 | ML integration extracts model features and optional ML artifacts. | Optional ML dependencies are probed before use; small input sets are reported. | ML integration coverage and overall tests. |
| 15 | Audio creates sonification/SAPF artifacts from model structure. | Audio backend and rendering settings are configurable; dependency issues are explicit. | Audio generation, SAPF, and integration tests. |
| 16 | Analysis computes statistics, framework comparisons, visualizations, GIFs, dashboards, and manifests. | `generate_animations` / `--no-animations` controls GIF output; stale framework outputs are scoped out. | Analysis tests, GridWorld manifest/GIF assertions, post-simulation tests. |
| 17 | Integration synthesizes cross-module outputs and meta-analysis evidence. | Reads available artifacts and reports missing handoffs. | Integration processor and functional tests. |
| 18 | Security scans maintained/generated code surfaces for policy issues. | Findings are structured; configured severity gates drive pass/fail behavior. | Security tests and Bandit gate. |
| 19 | Research produces experimental/research summaries from model inputs. | Works from parsed files and reports thin evidence separately. | Research functional and overall tests. |
| 20 | Website builds static documentation/report pages from pipeline artifacts. | Output path and source artifacts are configurable; minimal site generation is tested. | Website overall tests. |
| 21 | MCP registers and audits tool surfaces. | Module discovery is explicit; tool count regressions are checked. | MCP audit, functional, performance, and CI count assertion. |
| 22 | GUI exposes interactive constructors/editors around GNN models. | GUI dependencies are optional and reported when unavailable. | GUI functionality and integration tests. |
| 23 | Report assembles final Markdown/HTML/JSON reporting artifacts. | Report format/output config drives generation; partial evidence is documented. | Report format, functional, integration, and overall tests. |
| 24 | Intelligent analysis summarizes pipeline status and recommendations. | LLM use is configurable; non-LLM operation remains explicit. | Intelligent analysis coverage and overall tests. |

## Review Gates

Before publishing a hardening result, run:

```bash
uv run --extra dev ruff format --check src scripts
uv run --extra dev ruff check src scripts
uv run --extra dev mypy src --show-error-codes
uv run --extra dev bandit -r src -c pyproject.toml -q
uv run --extra dev python scripts/check_repo_terminology.py --strict
uv run --extra dev python scripts/check_maintained_doc_terms.py --strict
uv run --extra dev python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict
uv lock --check
julia --startup-file=no -e 'using RxInfer, ActiveInference, JSON, Distributions, StatsBase'
uv run --extra dev python -m pytest tests/pipeline/test_pomdp_gridworld_cross_framework.py -q --tb=short
uv run --extra dev python -m pytest tests/analysis/test_analysis_post_simulation.py tests/analysis/test_analysis_overall.py -q --tb=short
uv run --extra dev python -m pytest tests/execute/test_pymdp_contracts.py tests/execute/test_discrete_models_pymdp.py tests/visualization/test_visualization_matrices.py -q --tb=short
uv run --extra dev python -m pytest --collect-only tests/ -q --tb=no --ignore=tests/llm/test_llm_ollama.py --ignore=tests/llm/test_llm_ollama_integration.py
uv run --extra dev python -m pytest tests/ -q --tb=no --ignore=tests/llm/test_llm_ollama.py --ignore=tests/llm/test_llm_ollama_integration.py
```

Ollama integration tests are run separately when a local daemon and model are available.
