# Step 16: Analysis

## Architectural Mapping

**Orchestrator**: `src/gnn/16_analysis.py` (66 lines)
**Implementation Layer**: `src/gnn/analysis/`

## Module Description

This module provides comprehensive statistical analysis, performance profiling, and model evaluation capabilities for GNN models and pipeline components.


```
src/gnn/analysis/
├── __init__.py                    # Module initialization and exports
├── processor.py                   # Main analysis processor
├── analyzer.py                    # Statistical analysis functions
├── framework_extractors.py        # Per-framework result extraction
├── post_simulation.py             # Post-simulation analysis
├── interpretability.py            # Interpretability metrics
├── trace_analysis.py              # Execution-trace analysis
├── visualizations.py              # Shared visualization suite
├── generate_cross_model_report.py # Cross-model reporting
├── mcp.py                         # Model Context Protocol integration
└── <framework>/                   # Per-framework analyzers: rxinfer, pymdp,
                                   # activeinference_jl, jax, discopy, numpyro, pytorch
```

The RxInfer analyzer (`src/gnn/analysis/rxinfer/`) is the deepest of these and is documented in [RxInfer Analysis](#rxinfer-analysis) below.

## Agent Identity & Capabilities

# Analysis Module - Agent Scaffolding

## Module Overview

**Purpose**: Advanced statistical analysis, performance benchmarking, and complexity metrics calculation for GNN models

**Pipeline Step**: Step 16: Analysis (16_analysis.py)

**Category**: Statistical Analysis / Performance Evaluation

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-10-02

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
- **Cross-framework comparison** - uses whatever execution (Step 12) produced. `_extract_simulation_metrics` (in `framework_comparison.py`) prefers `simulation_data/simulation_results.json` (and other canonical JSON) before `execution_logs/*_results.json`, so backends that write full traces to `simulation_data/` (e.g. RxInfer) are not masked by sparse structured logs. DisCoPy: inline `simulation_data.analysis` / `parameters` from structured logs populate `circuit_info`; if still missing, `simulation_data/circuit_info.json` is merged when present. bnlearn structured logs populate `model_parameters` when vector traces are absent. If every run for a framework was skipped (`skipped: true` in the execution summary), logs INFO instead of WARNING for bnlearn. Otherwise missing data is reported as "[framework] No simulation data found". Optional Python backends require their declared setup extras; Julia coverage requires the pinned Julia environments before Step 12.
- **Degraded comparison columns** - every framework entry in
  `analyze_framework_outputs` carries an explicit `status` mirroring the Step 12
  summary vocabulary (`success` | `success_with_skips` | `skipped` | `failed`;
  `render_failed` for frameworks whose Step 11 render failed or whose render
  receipt no longer matches), plus `skipped_count` / `failed_count` /
  `status_reason` and matching status rows in the generated comparison report.
  One framework's degradation never crashes the comparison.
- **RxInfer analysis suite** - convergence diagnostics, per-factor belief recovery, per-model GIF animations with reproducibility manifests, an HTML dashboard, and cross-framework comparison. See [RxInfer Analysis](#rxinfer-analysis).

---

## RxInfer Analysis

`src/gnn/analysis/rxinfer/` consumes `rxinfer_simulation_v1` payloads written by Step 12 at `output/12_execute_output/<model>/rxinfer/simulation_data/simulation_results.json`, and writes to `output/16_analysis_output/rxinfer/`. The entry point is `generate_analysis_from_logs(execution_results_dir, output_dir, verbose=False)`.

Two properties of the input contract shape everything downstream: `variational_free_energy` and `vfe_per_iteration` are **per-iteration VFE traces** (length = inference iterations), the genuine convergence signal from variational message passing rather than per-step constants; and beliefs are **smoothed posteriors** from batch inference, not filtered online beliefs.

### Visualization suite

Matplotlib PNGs at dpi 300, one set per model, complementary to the optional Julia-native `Plots.jl` figures emitted at execute time: belief evolution, belief heatmap, observation/state traces, belief entropy, inference accuracy, action frequencies, belief convergence, belief trace, free energy, observations, and an EFE-per-action heatmap. Each plot is best-effort — one that cannot be produced (absent matplotlib, missing input arrays) is logged and skipped, never escalated to a step failure.

### Convergence diagnostics

Derived from the per-iteration VFE trace: VFE slope, convergence rate, and iterations-to-convergence.

### Per-factor belief recovery

Native `beliefs_by_agent` and `beliefs_by_factor` traces remain separate through the shared result adapter. Continuous agent views carry posterior means and covariances; they are never converted to categorical confidence. A missing component trace stays unavailable instead of being reconstructed from an unrelated result.

For an explicitly flattened categorical joint trace, `compute_per_factor_beliefs(data)` recovers marginals using the declared `model_parameters.state_factors` order. The product must match the joint width; C order makes the first factor slowest-varying. This recovery applies only to that declared joint representation.

An **empty dict signals structural absence, not failure**: no `state_factors` (flat models, or artifacts predating the key), no beliefs, or fewer than two factors of size > 1. Size-1 factors participate in the reshape but are omitted from the output. Genuine contract violations between renderer and analyzer — malformed descriptors, duplicate factor names, ragged belief rows, a size product contradicting the joint width, a timestep with no probability mass — raise `ValueError` rather than being quietly absorbed.

Multi-factor models additionally get per-factor belief-trajectory small-multiples.

### GIF animations and reproducibility manifests

`generate_gif_animation` produces one publication-style animated GIF per model (`<model>_rxinfer_animation.gif`): 2×3 panels covering beliefs (per-factor marginals when `state_factors` declares more than one factor), true vs inferred states, the Bayesian graph model, the VFE trace, the EFE-per-action heatmap, and the policy posterior. Each GIF carries a `.manifest.json` sidecar recording the GNN spec hash, Julia and RxInfer versions, seed, timesteps, inference iterations, and belief accuracy — enough to reproduce the artifact.

### Dashboard

`generate_dashboard` builds a single self-contained HTML dashboard over a directory of GIFs plus manifests, with a model-category filter, a state-size filter, and a side-by-side compare mode that shows any two models' animations and manifest statistics together.

### Strategy validation summary

`summarize_strategy_validation(data)` reads `runtime_metadata.model_kind` (defaulting to `flat` for payloads written before the field existed), asks the registered render-side `ModelStrategy` which validation fields it contributes via `get_validation_fields()`, and returns those fields that are actually present in the results `validation` dict. It is **loud on an unknown kind** (`ValueError`) but tolerant of a declared field being absent. The result is attached to the analysis as `validation_summary`, which keeps the analyzer's validation reporting in step with the renderer's strategies instead of hard-coding a field list.

### Cross-framework comparison

`run_cross_framework_comparison(gnn_file, output_dir)` renders one parsed GNN spec to RxInfer.jl, PyMDP, and ActiveInference.jl, runs all three, and writes `<model>_comparison.html` alongside a per-framework subdirectory of rendered scripts and raw results. The page carries a metrics table — including a per-framework status row giving the reason any framework did not succeed — and an animated belief-trajectory chart overlaying every framework's beliefs per hidden state over time, with play/pause and a step slider. The chart is a self-contained inline canvas script: no external assets, no network access.

---

## API Reference

### Public Functions

#### `process_analysis(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`
**Description**: Main analysis processing function called by orchestrator (16_analysis.py). Performs comprehensive statistical analysis, complexity metrics, and performance benchmarking.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to analyze
- `output_dir` (Path): Output directory for analysis results
- `verbose` (bool, optional): Enable verbose output (default: False)
- `**kwargs` (Any): Additional analysis options. Consumed keys: `generate_animations` / `no_animations` (GridWorld GIF animations; see the CLI Flags section). No `analysis_type`, `include_performance`, `include_complexity`, `include_quality`, or `benchmark_iterations` key is read by the current implementation.

**Returns**: `bool` - True if analysis succeeded, False otherwise

**Example**:
```python
from analysis import process_analysis
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    verbose=True,
)
```

### CLI Flags (`16_analysis.py`)

In addition to the standard pipeline flags (`--target-dir`, `--output-dir`, `--verbose`, etc.), `16_analysis.py` registers two step-specific flags via `create_standardized_pipeline_script`:

- `--advanced-stats` (store_true, default: off) — Registered CLI flag; **not consumed by the current `src/gnn/analysis/` implementation**, so it has no effect beyond parsed args.
- `--no-animations` (store_false into `generate_animations`, default: animations enabled) — Disable Step 16 GridWorld GIF animation artifacts. GridWorld animations are generated by default; pass `--no-animations` to skip them. The generated GIF/manifest is written via `analysis.visualizations.write_gridworld_analysis_manifest` when `generate_animations` is true.

**Example**:
```bash
uv run python src/gnn/16_analysis.py --target-dir input/gnn_files --output-dir output --advanced-stats
uv run python src/gnn/16_analysis.py --target-dir input/gnn_files --output-dir output --no-animations
```

#### `perform_statistical_analysis(file_path: Path, verbose: bool = False) -> Dict[str, Any]`
**Description**: Perform comprehensive statistical analysis on a GNN file.

**Parameters**:
- `file_path` (Path): Path to the GNN file to analyze
- `verbose` (bool, optional): Enable verbose output (default: False)

**Returns**: `Dict[str, Any]` - Statistical analysis results with:
- `variable_statistics` (Dict[str, Any]): Statistics on variables
- `connection_statistics` (Dict[str, Any]): Statistics on connections
- `section_statistics` (Dict[str, Any]): Statistics on model sections
- `distributions` (Dict[str, Any]): Variable distributions
- `correlations` (Dict[str, Any]): Variable correlations

#### `calculate_complexity_metrics(model_data: Dict[str, Any], variables: List[Dict[str, Any]] = None, connections: List[Dict[str, Any]] = None) -> Dict[str, Any]`
**Description**: Calculate various complexity metrics for GNN models.

**Parameters**:
- `model_data` (Dict[str, Any]): Parsed GNN model data
- `variables` (List[Dict[str, Any]], optional): Model variables (extracted if not provided)
- `connections` (List[Dict[str, Any]], optional): Model connections (extracted if not provided)

**Returns**: `Dict[str, Any]` - Complexity metrics with:
- `cyclomatic_complexity` (float): Cyclomatic complexity score
- `cognitive_complexity` (float): Cognitive complexity score
- `structural_complexity` (float): Structural complexity score
- `maintainability_index` (float): Maintainability index (0-100)
- `technical_debt` (float): Technical debt score

**Returns**: Dictionary with complexity metrics (cyclomatic, cognitive, structural)

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
- `ANALYSIS_PERFORMANCE_MODE` - Performance analysis mode ("fast", "comprehensive") *(reserved/illustrative — not consumed by the current implementation)*
- `ANALYSIS_TIMEOUT` - Maximum analysis time per model (default: 300 seconds) *(reserved/illustrative — not consumed by the current implementation)*

### Configuration Files
- `analysis_config.yaml` - Custom analysis parameters and thresholds *(reserved/illustrative — not tracked by the current implementation)*

### Default Settings
`DEFAULT_COMPLEXITY_THRESHOLDS` is not a defined constant in the implementation; complexity thresholds are computed inline within `calculate_complexity_metrics`. The block below is illustrative only:
```python
# Illustrative — thresholds are computed inline, not read from this constant
DEFAULT_COMPLEXITY_THRESHOLDS = {
    "cyclomatic_complexity": {"low": 10, "medium": 20, "high": 50},
    "cognitive_complexity": {"low": 5, "medium": 15, "high": 35},
    "structural_complexity": {"low": 100, "medium": 500, "high": 1000},
}
```

---

## Usage Examples

### Basic Usage
```python
from gnn.analysis.processor import process_analysis

success = process_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/16_analysis_output"),
    verbose=True,
)
```

### Statistical Analysis
```python
from gnn.analysis.analyzer import perform_statistical_analysis

stats = perform_statistical_analysis(Path("input/gnn_files/some_model.gnn"), verbose=True)
print(f"Variable count: {stats['variable_statistics']['count']}")
print(f"Connection count: {stats['connection_statistics']['count']}")
```

### Complexity Assessment
```python
from gnn.analysis.analyzer import calculate_complexity_metrics

metrics = calculate_complexity_metrics(parsed_model)
print(f"Cyclomatic complexity: {metrics['cyclomatic_complexity']}")
print(f"Maintainability index: {metrics['maintainability_index']}")
```

---

## Output Specification

### Output Products
- `analysis_results.json` - Comprehensive statistical analysis
- `{model}_post_simulation_analysis.json` - Post-simulation analysis results
- `analysis_summary.md` - Human-readable analysis summary report
- `cross_model_comparison_report.md` - Cross-model comparison report
- `analysis_processing_summary.json` - Pipeline step summary

### Output Directory Structure
```
output/16_analysis_output/
├── analysis_results.json
├── model_name_post_simulation_analysis.json
├── analysis_summary.md
├── cross_model_comparison_report.md
├── analysis_processing_summary.json
├── pymdp_visualizations/              # All PyMDP visualizations
│   └── {model_name}/
│       ├── discrete_states.png
│       ├── belief_evolution.png
│       ├── performance_metrics.png
│       └── action_sequence.png
├── rxinfer/                           # RxInfer analysis suite
│   ├── {model_name}_rxinfer_*.png     # Per-model visualization set
│   ├── {model_name}_rxinfer_animation.gif
│   ├── {model_name}_rxinfer_animation.manifest.json
│   └── {model_name}_comparison.html   # When cross-framework comparison is run
└── comprehensive_visualizations/
```

---

## Performance Characteristics

Measure duration and memory with the same selected models, configuration and runtime when comparing implementations. Current acceptance receipts live in [the verification ledger](../../development/verification_2026_10_01.json). A per-framework timing aggregate is descriptive; different corpora or environments do not establish a speed comparison.

---

## Error Handling

### Graceful Degradation
- **No scipy**: Simplified statistical analysis using numpy
- **No matplotlib**: Text-based statistical reports
- **Large models**: Sampling-based analysis with warnings

### Error Categories
1. **Statistical Errors**: Invalid data types or missing values
2. **Complexity Errors**: Model structure too complex for analysis
3. **Performance Errors**: Timeout or resource exhaustion

---

## Integration Points

### Orchestrated By
- **Script**: `16_analysis.py` (Step 16)
- **Function**: `process_analysis()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests.test_analysis_integration.py` - Integration tests
- `report.generator` - Scans `16_analysis_output` via the uniform all-steps census (no dedicated Step-16 data path)

### Data Flow
```
input/gnn_files (re-parsed) + 12_execute_output (execution results) → Analysis → output/16_analysis_output/
                                              ↓
        Step 20 (website) loads analysis JSONs; Step 23 census-includes this dir
```

---

## Testing

### Test Files
- `tests/analysis/test_analysis_overall.py` - Module-level tests
- `tests/analysis/test_analysis_post_simulation.py` - Post-simulation analysis tests
- `tests/analysis/test_analysis_extraction.py` - Result extraction tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/analysis/ --cov=gnn.analysis --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Statistical analysis with various model sizes
2. Complexity metric calculation accuracy
3. Performance benchmarking under load
4. Error handling with malformed data

---

## MCP Integration

### Tools Registered

Registered in `register_tools` (`src/gnn/analysis/mcp.py`):

- `process_analysis` - Run Step 16 over a directory
- `get_analysis_results` - Read saved analysis results
- `compute_complexity_metrics` - Compute complexity metrics for a model
- `list_analysis_tools` - List registered analysis tools

### MCP File Location
- `src/gnn/analysis/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Analysis fails on large models
**Symptom**: Analysis times out or runs out of memory  
**Cause**: Model too complex for comprehensive analysis  
**Solution**: 
- Process models one at a time (there is no analysis-type flag; see the CLI Flags section)
- Increase system memory or use sampling

#### Issue 2: Complexity metrics return zero
**Symptom**: Complexity calculations return zero or invalid values  
**Cause**: Model structure not properly extracted or missing components  
**Solution**:
- Verify GNN processing (step 3) completed successfully
- Check that model has variables and connections
- Use `--verbose` flag for detailed extraction logs

#### Issue 3: Framework comparison fails
**Symptom**: Cross-framework comparison reports errors  
**Cause**: Execution results (step 12) not available or incomplete  
**Solution**:
- Ensure execution step (12) completed successfully
- Verify framework outputs exist in execution results
- Check execution results format matches expected structure

---

## Version History

### Current package version

See [pyproject.toml](../../../pyproject.toml).

**Features**:
- Statistical analysis
- Complexity metrics calculation
- Performance benchmarking
- Model comparison
- Framework output analysis

**Verification boundaries**:
- Numerical comparisons require compatible, source-bound results; missing metadata produces an unavailable comparison.
- Live optional-backend acceptance is recorded separately from dependency-light tests.

### Roadmap
- **Next Version**: Enhanced visualization of analysis results
- **Future**: Real-time analysis dashboard

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Execute Module](../../../src/gnn/analysis/../execute/AGENTS.md)
- [Analysis Module](../../../src/gnn/analysis/../analysis/README.md)

### External Resources
- [NetworkX Documentation](https://networkx.org/)
- [NumPy Documentation](https://numpy.org/docs/)
- [SciPy Documentation](https://scipy.org/)

---

**Last Updated**: 2026-10-02
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/analysis/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/analysis/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/analysis/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/analysis/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/analysis](../../../src/gnn/analysis)
