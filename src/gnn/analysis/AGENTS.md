# Analysis Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced statistical analysis, performance benchmarking, and complexity metrics calculation for GNN models

**Pipeline Step**: Step 16: Analysis (16_analysis.py)

**Category**: Statistical Analysis / Performance Evaluation

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-25

---

## Core Functionality

### Primary Responsibilities
1. Perform comprehensive statistical analysis on GNN model structures
2. Calculate complexity metrics and maintainability indices
3. Generate performance benchmarks and comparison reports
4. Extract and analyze variable distributions and correlations
5. Provide technical debt assessment and optimization recommendations
6. **Generate ALL PyMDP visualizations** from execution raw data (moved from Execute step)

### Key Capabilities
- Statistical analysis of model variables and connections
- Complexity metrics calculation (cyclomatic, cognitive, structural)
- Performance benchmarking and profiling
- Model comparison and differential analysis
- Distribution analysis and correlation studies
- **PyMDP Visualization** - belief evolution, state sequences, performance metrics plots
- **Cross-framework comparison** - uses whatever execution (Step 12) produced. `_extract_simulation_metrics` (in `analyzer.py`) prefers `simulation_data/simulation_results.json` (and other canonical JSON) before `execution_logs/*_results.json`, so backends that write full traces to `simulation_data/` (e.g. RxInfer) are not masked by sparse structured logs. DisCoPy: inline `simulation_data.analysis` / `parameters` from structured logs populate `circuit_info`; if still missing, `simulation_data/circuit_info.json` is merged when present. bnlearn structured logs populate `model_parameters` when vector traces are absent. If every run for a framework was skipped (`skipped: true` in the execution summary), logs INFO instead of WARNING for bnlearn. Otherwise missing data is reported as "[framework] No simulation data found". Python backends are in core `uv sync`; Julia coverage needs Julia + packages installed, then re-run Step 12.
- **Kronecker-factorized JAX (MAJ-02)** - `extract_jax_data` dispatches on
  schema: `jax_kronecker_factorized_v1` payloads (top-level, nested
  `simulation_data`, or implementation-directory files) are extracted by
  `extract_jax_kronecker_data` into per-factor fields — beliefs/states/
  observations/actions per factor, per-step total EFE (sum over factors),
  factorised policy, validation, and model parameters with
  `joint_state_space_size` / `joint_materialized: False`. pymdp-compatible
  JAX payloads keep the historical path.
- **GridWorld animations** - current PyMDP, RxInfer.jl, and ActiveInference.jl
  schemas (plus the ungated NumPyro and PyTorch runner schemas, which the
  manifest lists alongside the strict three) emit belief GIFs, 3x3 state
  trajectory GIFs, a cross-framework trajectory GIF, and
  `cross_framework/gridworld_analysis_manifest.json`.

- **Static complexity estimation (`complexity/` subpackage, wave 8)** — pure-stdlib,
  execution-free per-backend bounds: `gnn.analysis.complexity.estimate_model_complexity`
  emits the `gnn.complexity_estimate/v1` receipt (structure stats, model kinds,
  `per_backend` rows in fixed `BACKEND_ORDER` with `applicable` flags);
  see [complexity/AGENTS.md](complexity/AGENTS.md) and
  [complexity/README.md](complexity/README.md). Distinct from the per-file
  complexity metrics (`calculate_complexity_metrics`, cyclomatic/cognitive/structural),
  which operate on source files rather than parsed models.

---

## API Reference

### Public Functions

#### `process_analysis(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool | int`
**Description**: Main analysis processing function called by orchestrator (16_analysis.py). Performs statistical analysis, complexity metrics, performance benchmarks, post-simulation framework analysis, and visualizations.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for analysis results
- `verbose` (bool): Enable verbose output (default: False)
- `generate_animations` (bool, optional): Generate current-schema GridWorld GIF
  artifacts (default: True; CLI: `--no-animations` disables this). This is
  the canonical programmatic key. Callers passing the inverse flag
  `no_animations` have it normalized as the inverse, and conflicts with
  `generate_animations` are rejected.
- `**kwargs`: Additional pipeline options (unused kwargs are ignored)

**Returns**: `True` if analysis artifacts were produced, `2` when there is no
input or other warning-only recovery, and `False` for hard failures.


**Example**:
```python
from gnn.analysis import process_analysis
from pathlib import Path

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    verbose=True,
)
```

#### `perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Perform comprehensive statistical analysis on a GNN file.

**Parameters**:
- `file_path` (Path): Path to the GNN file to analyze
- `verbose` (bool, optional): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Statistical analysis results with:
- `variable_statistics` / `connection_statistics` / `section_statistics` (Dict[str, Any])
- `distributions` and `correlations` (Dict[str, Any])
- `file_path`, `file_name`, `file_size`, `line_count`, `analysis_timestamp`

Raises `RuntimeError` if the file cannot be analyzed.

#### `calculate_complexity_metrics(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Calculate various complexity metrics for a GNN file.

**Parameters**:
- `file_path` (Path): Path to the GNN file to analyze
- `verbose` (bool, optional): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Complexity metrics with:
- `cyclomatic_complexity` (float): Cyclomatic complexity score
- `cognitive_complexity` (float): Cognitive complexity score
- `structural_complexity` (float): Structural complexity score
- `maintainability_index` (float): Maintainability index (0-100)
- `technical_debt` (float): Technical debt score

Raises `RuntimeError` if metrics cannot be computed.

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations and statistical analysis
- `pandas` - Data manipulation and analysis
- `scipy` - Advanced statistical functions

### Optional Dependencies
- `matplotlib` - Statistical visualization (recovery: text-based reports)
- `seaborn` - Enhanced statistical plots (recovery: matplotlib)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing patterns
- `pipeline.config` - Pipeline configuration management

---

## Configuration

### Environment Variables

None dedicated to this module. Analysis behavior is configured through
`process_analysis()` kwargs (e.g. `analysis_type`, `benchmark_iterations`,
`generate_animations`) and `input/config.yaml` pipeline settings.

### Default Settings

Complexity thresholds and benchmark parameters are defined in
`analysis/analyzer.py`; see the `perform_statistical_analysis` and
`calculate_complexity_metrics` functions above.

---

## Usage Examples

### Basic Usage
```python
from gnn.analysis.processor import process_analysis
from pathlib import Path

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    verbose=True,
)
```

### Statistical Analysis
```python
from gnn.analysis.analyzer import perform_statistical_analysis

stats = perform_statistical_analysis(Path("models/my_model.md"))
print(f"Variable count: {len(stats['variable_statistics'])}")
```

### Complexity Assessment
```python
from gnn.analysis.analyzer import calculate_complexity_metrics

metrics = calculate_complexity_metrics(Path("models/my_model.md"))
print(f"Cyclomatic complexity: {metrics['cyclomatic_complexity']}")
print(f"Maintainability index: {metrics['maintainability_index']}")
```

---

## Output Specification

### Output Products

`process_analysis` writes to the step output directory:

- `analysis_results.json` - Full step results (statistical, complexity, benchmarks, comparisons)
- `analysis_summary.md` - Human-readable analysis report
- `cross_model_comparison_report.md` - Cross-framework comparison (when execution data exists)
- `{model}_post_simulation_analysis.json` - Per-model post-simulation analysis (in the cross-framework analysis subdirectory)
- Visualization directories (`comprehensive_visualizations/`, framework GIFs and PyMDP visualizations, `cross_framework/gridworld_analysis_manifest.json` when animations are enabled)

---

## Error Handling

- **No input**: returns exit code `2` with a warning (not a failure)
- **Missing execution summary**: logs a warning and skips post-simulation analysis
- **Malformed data**: per-file exceptions are collected in `results["errors"]`; the step continues with other files
- **Animation flag conflict**: `generate_animations` conflicting with `no_animations` aborts with `False`

## Integration Points

### Imported By
- `tests/analysis/` - Module-level analysis tests
- `report.generator` - Report generation uses analysis results

### Data Flow
```
GNN Files → Analysis → Statistical Reports → Model Comparisons → Optimization Recommendations
```

---

## Testing

### Test Files
- `tests/analysis/test_analysis_overall.py` - Module-level tests
- `tests/analysis/test_analysis_post_simulation.py` - Post-simulation analysis tests
- `tests/analysis/test_analysis_extraction.py` - Result extraction tests
- `tests/analysis/test_framework_common.py` - Shared framework-common helper tests
- `tests/analysis/test_flat_payload_analyzer.py` - Shared flat-payload analyzer engine tests

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/analysis/ \
    --cov=src/gnn/analysis --cov-report=term-missing
```

### Key Test Scenarios
1. Statistical analysis with various model sizes
2. Complexity metric calculation accuracy
3. Performance benchmarking under load
4. Error handling with malformed data

---

## MCP Integration

### Tools Registered

`analysis/mcp.py` `register_tools` registers four tools:

- `process_analysis` - Run statistical and complexity analysis on GNN files in a directory
- `get_analysis_results` - Read saved analysis JSON results from a previous run
- `compute_complexity_metrics` - Compute complexity metrics for GNN content supplied as a string
- `list_analysis_tools` - Report available analysis tools and capabilities (honest availability probe — reports measured availability; never asserts availability unconditionally)

### MCP File Location
- `src/gnn/analysis/mcp.py` - MCP tool registrations

---


## Shared Composability Helpers

### `framework_common.py`

Single source of truth for framework-name normalization, path inference, and
current-schema simulation-results discovery. Consumed by `processor.py` (and
available to all framework analyzers):

- `FRAMEWORK_DIR_NAMES` — frozenset of the 8 analyzed pipeline frameworks
  (incl. `bnlearn`, which is rendered+executed but has no analyzer).
- `SCHEMA_GATED_FRAMEWORKS` — frozenset `{pymdp, rxinfer, activeinference_jl}`.
- `CURRENT_SIMULATION_SCHEMAS` — frozenset of `*_simulation_v1` schema strings.
- `normalize_framework_name(framework) -> str` — `"ActiveInference.jl"` → `"activeinference_jl"`.
- `model_name_from_path(path) -> str` — infers the model name from the path
  segment preceding a framework segment.
- `framework_from_path(path) -> str | None` — returns the framework dir name
  found in a path, or `None`.
- `iter_current_schema_results(execution_dir, pattern) -> list[tuple[Path, dict]]` —
  discovers current-schema `simulation_results.json` payloads; schema-gated
  frameworks must match `CURRENT_SIMULATION_SCHEMAS`, others accepted as-is.
- `resolve_execution_dir(output_dir) -> Path` — resolves the Step 12 execution
  output directory via the shared `pipeline.config.resolve_step_output_dir` helper (single fallback policy: caller-supplied directory on standalone use).
- `load_execution_summary(execution_dir) -> tuple[Path, dict | None]` — prefers
  `summaries/execution_summary.json` then root; returns `None` on missing/unreadable.
- `filter_paths_by_scope(path, framework, allowed_frameworks, allowed_model_names) -> bool`.

### `flat_payload_analyzer.py`

Shared analyzer engine for PyTorch/NumPyro flat-payload simulation results.
Each framework's `analyzer.py` binds a `FlatPayloadSpec` (framework name, file
patterns, analysis filename, plot labels, bar color) and re-exports
`generate_analysis_from_logs` / `_generate_plots` — the public call sites
(processor's importlib discovery, `test_numpyro_pytorch_analyzers.py`) are
unchanged. Exports: `FlatPayloadSpec`, `compute_flat_payload_metrics` (pure),
`discover_result_files`, `generate_analysis_from_logs`.

---

## Module Coverage

Per-file contract for `processor.py`, `viz_plots.py`, `framework_extractors.py`,
and `viz_dashboard.py` (the `process_analysis` API itself is specified in the
[API Reference](#api-reference); the shared-helper modules are covered under
Shared Composability Helpers). Line anchors refer to the current tree.

### `processor.py`

**Role**: Step 16 entry point. `process_analysis` (processor.py:282) runs the
full step pipeline and writes `analysis_results.json` (processor.py:713) and
`analysis_summary.md` (processor.py:722).

**Public surface** (beyond `process_analysis`):

- `aggregate_simulation_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]`
  (processor.py:205) — collects `execution_time` / `free_energy_final` /
  `steps_completed` metric lists and the set of frameworks used across
  post-simulation result rows.
- `generate_summary_statistics(aggregated_data: Dict[str, Any]) -> Dict[str, Any]`
  (processor.py:247) — mean/std/min/max/count per metric; consumes the
  `aggregate_simulation_results` output and populates
  `results["overall_statistics"]` (processor.py:476-482).
- `convert_numpy_types(obj: Any) -> Any` (processor.py:742) — `json.dump`
  default for numpy scalars/arrays, sets, `Path`s, and unknown objects;
  re-exported from `gnn.analysis` (`analysis/__init__.py:79`).

**Private helpers (contract-relevant)**:

- `_coerce_bool_flag` (processor.py:50) and `_normalize_generate_animations`
  (processor.py:65) — animation-flag normalization: `generate_animations` is
  canonical, `no_animations` is only a compatibility inverse accepted when the
  canonical key is absent, and a conflicting pair raises `ValueError`, which
  `process_analysis` converts to `False` (processor.py:326-329).
- `_scope_from_execution_summary` (processor.py:111) — derives current-run
  framework/model scope from the Step 12 summary; successful details carrying
  a result pointer narrow the framework set.
- `_filter_execution_summary` (processor.py:175) — deep-copied summary
  limited to current-run frameworks before empirical visualization.

**Maintenance contract**:

- Per-framework analyzer dispatch table `_FRAMEWORK_ANALYZERS`
  (processor.py:492-500) lists pymdp, activeinference_jl, discopy, jax,
  rxinfer, pytorch, numpyro. Each is imported as
  `importlib.import_module(".{module_key}.analyzer", package="analysis")`
  (processor.py:512-514) and called as
  `generate_analysis_from_logs(execution_dir, fw_output_dir, verbose)`
  (processor.py:518-520). Adding a framework requires a row here plus
  `analysis/<framework>/analyzer.py`; bnlearn is rendered/executed but has no
  analyzer and stays absent.
- Cross-module calls resolve through re-exports: `analyze_execution_results`
  (processor.py:422, called at processor.py:432),
  `generate_unified_framework_dashboard` (processor.py:588, called at
  processor.py:644), and `visualize_all_framework_outputs` (processor.py:589,
  called at processor.py:595) come from `post_simulation`;
  `_current_schema_visualization_data` (processor.py:591) from
  `visualizations` (re-exported from `viz_schema`);
  `write_gridworld_analysis_manifest` (processor.py:693) from
  `visualizations` (re-exported from `viz_manifest`).

### `viz_plots.py`

**Role**: Per-model execution-output plotting (extracted from
`visualizations.py`). Consumes `viz_base` (`np`, `plt`, `safe_savefig`),
`viz_schema` (schema gating), `viz_animations` (GridWorld suite), and
`viz_dashboard` (comparison charts) (viz_plots.py:18-34).

**Public surface**:

- `visualize_all_framework_outputs(execution_dir: Path, output_dir: Path, logger_instance: Optional[logging.Logger] = None, allowed_frameworks: Optional[set[str]] = None, allowed_model_names: Optional[set[str]] = None, generate_animations: bool = True) -> List[str]`
  (viz_plots.py:39) — discovers `*_results.json` (excluding `simulation_data`
  subtrees, viz_plots.py:87-89) and `*simulation_results.json`
  (viz_plots.py:132), merges entries per `(framework, model)` key, and
  schema-gates pymdp/rxinfer/activeinference_jl against
  `CURRENT_VISUALIZATION_SCHEMAS` (viz_plots.py:97-102, 151-156). Writes:
  - `{model}_{framework}_free_energy.png`, `{model}_{framework}_vfe_vs_efe.png`,
    `{model}_{framework}_observations.png` under
    `output_dir.parent / <framework>` (viz_plots.py:321, 340-389);
  - belief heatmaps and action analysis are deliberately NOT produced here —
    the per-framework analyzers already emit richer versions
    (viz_plots.py:324-327); the exported builders remain for direct reuse;
  - when more than one framework has data:
    `cross_framework_comparison.png`, `efe_convergence_comparison.png`,
    `confidence_comparison.png`, `framework_radar.png`
    (viz_plots.py:397-444) and, when `generate_animations`, the GridWorld
    animation suite (viz_plots.py:446-453).
- Scalar plot builders — each returns the saved path, raises `ValueError` on
  empty input, and saves through `viz_base.safe_savefig`:
  - `generate_belief_heatmaps(beliefs: List[List[float]], output_path: Path, title: str = "Belief State Evolution Heatmap") -> str`
    (viz_plots.py:459) — heatmap + per-state trajectory pair; needs at least
    two timesteps.
  - `generate_action_analysis(actions: List[int], output_path: Path, title: str = "Action Selection Analysis") -> str`
    (viz_plots.py:516) — histogram, sequence, transition matrix.
  - `generate_free_energy_plots(free_energy: List[float], output_path: Path, title: str = "Free Energy Dynamics") -> str`
    (viz_plots.py:629) — 2x2 panel: evolution with min-EFE overlay (handles
    per-policy 2D input), selected-EFE distribution, per-step change,
    rolling-variance convergence.
  - `generate_vfe_vs_efe_plot(vfe: List[float], efe: List[Any], output_path: Path, title: str = "Variational vs Expected Free Energy") -> str`
    (viz_plots.py:821) — dual-axis VFE vs min-EFE per step.
  - `generate_observation_analysis(observations: List[int], output_path: Path, title: str = "Observation Analysis") -> str`
    (viz_plots.py:886) — frequency + sequence.

**Maintenance contract**: public re-export chain is `visualizations.py`
(viz_plots import at visualizations.py:35-42) → `post_simulation.py`
(post_simulation.py:97-108) → `analysis/__init__`; signature changes must
update all three. Split-smoke test for the split:
`tests/analysis/test_analysis_split_smoke.py`.

### `framework_extractors.py`

**Role**: Per-framework `extract_*_data(execution_result) -> Dict[str, Any]`
normalizers — the single source for turning raw Step 12 results into analysis
fields. `post_simulation.analyze_execution_results` dispatches on the
framework name (post_simulation.py:188-206).

**Schema constants**:

- `CURRENT_SIMULATION_SCHEMAS` (framework_extractors.py:20) — framework →
  `*_simulation_v1` mapping (pytorch, numpyro, pymdp, rxinfer,
  activeinference_jl).
- `KRONECKER_FACTORIZED_SCHEMA = "jax_kronecker_factorized_v1"`
  (framework_extractors.py:458).

**Extractors**:

- `extract_pymdp_data` (framework_extractors.py:96) — strict
  `pymdp_simulation_v1` (top-level payload, nested `simulation_data`, or
  impl-dir files); counts `visualizations/*.{png,svg}`
  (framework_extractors.py:135-142); missing or wrong-schema payload sets
  `extraction_error` (framework_extractors.py:147-152).
- `extract_rxinfer_data` (framework_extractors.py:187) — a schema hit returns
  the normalized payload; otherwise the collected-file fallback maps
  `efe_history` → `free_energy` (framework_extractors.py:244-246).
- `extract_activeinference_jl_data` (framework_extractors.py:273) — schema hit
  → normalizer; else `simulation_results.csv` (impl dir or
  `activeinference_outputs_*`, framework_extractors.py:315-322) or JSON
  fallback; returns the full Active-Inference field set (A/B/C/D matrices,
  precisions, VFE, information gain, pragmatic value;
  framework_extractors.py:418-453).
- `extract_jax_data` (framework_extractors.py:525) — dispatches to
  `extract_jax_kronecker_data` (framework_extractors.py:461) on the Kronecker
  schema (top-level, nested, or impl-dir probe); everything else follows the
  pymdp-compatible historical path (framework_extractors.py:547).
- `extract_discopy_data` (framework_extractors.py:550) — reads
  `discopy_execution_report.json` from three candidate locations
  (framework_extractors.py:570-577); diagrams come from successful
  `diagram_validation` executions (framework_extractors.py:593-598).
- `extract_pytorch_data` (framework_extractors.py:670) and
  `extract_numpyro_data` (framework_extractors.py:677) — shared
  `_extract_schema_aware_data` (framework_extractors.py:629): schema-gated
  dispatch, `efe_history` → `expected_free_energy`
  (framework_extractors.py:665-666), ungated schema-less fallback (nested
  `simulation_data` holding beliefs/actions/observations, or the execution
  result itself, framework_extractors.py:658-663).

**Private helpers**: `_normalise_current_simulation_payload`
(framework_extractors.py:29) — canonical field mapping shared by the pymdp /
rxinfer / activeinference_jl / pytorch / numpyro paths; pymdp passes
`fallback_top_level=False` because its schema stores data only in by-factor
maps (framework_extractors.py:37-39; pinned by
`tests/render/test_jax_factorized_pipeline.py`).
`_load_current_schema_from_impl_dir` (framework_extractors.py:70) — ordered
impl-dir probe for a schema-stamped payload.

**Maintenance contract**: `post_simulation.py` re-exports all extractor
functions (framework_extractors import at post_simulation.py:69-78; listed in
its `__all__` at post_simulation.py:28-35). `analysis/__init__` re-exports the
pymdp / rxinfer / activeinference_jl / jax / jax_kronecker / discopy extractors
via its `post_simulation` import (`analysis/__init__.py:64-69`); the pytorch
and numpyro extractors are importable from
`gnn.analysis.framework_extractors` and `gnn.analysis.post_simulation`, not
from `gnn.analysis` directly. Adding an extractor: extend
`framework_extractors.py`, the `post_simulation` import and `__all__`, and the
package-level `analysis/__init__` import when the extractor should be public.
All extractors accept one execution-result dict and never raise for absent
data: pymdp signals `extraction_error`; the others return empty lists.

### `viz_dashboard.py`

**Role**: Cross-framework dashboards and comparison charts. Consumed directly
by `visualizations.py` (viz_dashboard import at visualizations.py:24-30) and
by `viz_plots.visualize_all_framework_outputs` (viz_plots.py:24-29).

**Public surface**:

- `generate_unified_framework_dashboard(framework_data: Dict[str, Dict[str, Any]], output_dir: Path, model_name: str = "Active Inference Model") -> List[str]`
  (viz_dashboard.py:28) — up to three artifacts under `output_dir`:
  `unified_belief_comparison.png` when two or more frameworks have beliefs
  (viz_dashboard.py:106, 171), `unified_action_efe_comparison.png` when
  actions or EFE exist (viz_dashboard.py:178, 261), and
  `unified_entropy_comparison.png` (viz_dashboard.py:267, 312). Called from
  processor.py:644 with `cross_framework/unified_dashboard`.
- `generate_cross_framework_comparison(framework_data: Dict[str, Dict[str, Any]], output_path: Path) -> str`
  (viz_dashboard.py:320) — execution-time / steps / success-rate bars,
  aggregated by unique framework name (viz_dashboard.py:333-403); raises
  `ValueError` when nothing aggregated (viz_dashboard.py:375-376).
- `generate_efe_convergence_comparison(framework_data: Dict[str, Dict[str, Any]], output_path: Path) -> List[str]`
  (viz_dashboard.py:449) — raw + running-mean EFE overlay; returns `[]`
  unless two or more frameworks provide an EFE series
  (viz_dashboard.py:500-502).
- `generate_confidence_comparison(framework_data: Dict[str, Dict[str, Any]], output_path: Path) -> List[str]`
  (viz_dashboard.py:551) — confidence + uncertainty panels; returns `[]`
  unless two or more frameworks have confidence data
  (viz_dashboard.py:599-601).
- `generate_framework_radar(exec_summary_path: Path, framework_data: Dict[str, Dict[str, Any]], output_path: Path) -> List[str]`
  (viz_dashboard.py:643) — five-axis radar (Speed, Data Richness, Belief
  Quality, Timesteps, Validation; viz_dashboard.py:737-743) built from
  `execution_summary.json` details plus collected simulation data
  (viz_dashboard.py:671-731); returns `[]` for fewer than two frameworks
  (viz_dashboard.py:733-734).

**Maintenance contract**: the EFE/confidence/radar comparisons return `[]`
when matplotlib is unavailable (`MATPLOTLIB_AVAILABLE` from `viz_base`,
viz_dashboard.py:17-22); all saves go through `viz_base.safe_savefig`;
framework names normalize via `viz_schema._normalize_framework_name`
(viz_dashboard.py:23). Re-export chain: `visualizations.py` →
`post_simulation.py` → `analysis/__init__`.

---

## Troubleshooting

### Common Issues

#### Issue 1: Analysis fails on large models
**Symptom**: Analysis is slow or memory-heavy  
**Cause**: Model too complex for comprehensive analysis  
**Solution**: 
- Process models individually instead of batch
- Increase system memory or use sampling

#### Issue 2: Complexity metrics return zero
**Symptom**: Complexity calculations return zero or invalid values  
**Cause**: Model structure not properly extracted or missing components  
**Solution**:
- Verify GNN processing (step 3) completed successfully
- Check that model has variables and connections
- Use `--verbose` flag for detailed extraction logs

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Statistical analysis
- Complexity metrics calculation
- Performance benchmarking
- Model comparison
- Framework output analysis
- `framework_common.py` — shared framework-name normalization, path inference, and current-schema simulation-results discovery (dedupes processor.py / visualizations.py copies; now includes bnlearn in the framework dir set so bnlearn results are discoverable by the analysis scope)
- `flat_payload_analyzer.py` — shared analyzer engine for PyTorch/NumPyro flat-payload simulation results (frozen `FlatPayloadSpec` + pure `compute_flat_payload_metrics` + shared discovery/plots); each framework's `analyzer.py` is now a thin spec binding
- `mcp.list_analysis_tools_mcp` honest-availability probe (the unconditional `"available": True` reply was removed; availability is measured)
- `visualizations.py` matplotlib routed through `viz_base.safe_savefig` (single save/close/error path; 13 duplicated boilerplate sites consolidated)

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced visualization of analysis results
- **Future**: Real-time analysis dashboard

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Execute Module](../execute/AGENTS.md)
- [Analysis Module](../analysis/README.md)

### External Resources
- [NetworkX Documentation](https://networkx.org/)
- [NumPy Documentation](https://numpy.org/docs/)
- [SciPy Documentation](https://scipy.org/)

---

**Last Updated**: 2026-09-25
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
