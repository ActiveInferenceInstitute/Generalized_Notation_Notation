# Intelligent Analysis Module - Agent Scaffolding

## Module Overview

**Purpose**: AI-powered analysis of pipeline execution results with failure root cause identification, performance bottleneck detection, per-step flag analysis, and executive report generation

**Pipeline Step**: Step 24: Intelligent Analysis (24_intelligent_analysis.py)

**Category**: Pipeline Intelligence / Executive Reporting

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Module Structure

```
intelligent_analysis/
├── __init__.py       # Public API exports, module utilities, tool checks
├── AGENTS.md         # This documentation
├── processor.py      # Core analysis processing logic and report generation
├── analyzer.py       # IntelligentAnalyzer class, health scoring, pattern detection
├── history.py        # Run-history snapshots, deltas, trend classification
├── remediation.py    # ContractViolation fix suggestions (auxiliary)
└── mcp.py            # MCP tool registrations
```

## Key Components

### Processor (`processor.py`)

- `process_intelligent_analysis()` - Main entry point for pipeline step
- `analyze_pipeline_summary()` - Analyze overall pipeline execution
- `analyze_individual_steps()` - Per-step analysis with flag detection
- `generate_executive_report()` - Generate comprehensive reports
- `identify_bottlenecks()` - Find performance issues
- `extract_failure_context()` - Extract failure details
- `generate_recommendations()` - Generate improvement suggestions
- `StepAnalysis` - Data class for step analysis results

### Analyzer (`analyzer.py`)

- `IntelligentAnalyzer` - Main analyzer class with LLM integration
- `AnalysisContext` - Context container for analysis state
- `calculate_pipeline_health_score()` - Compute overall health metrics
- `classify_failure_severity()` - Categorize failure types
- `detect_performance_patterns()` - Identify performance trends
- `generate_optimization_suggestions()` - LLM-powered optimization advice

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Pipeline Analysis | Enabled | Full pipeline execution analysis |
| Failure Root Cause | Enabled | Deep failure analysis and causes |
| Performance Optimization | Enabled | Bottleneck identification |
| LLM-Powered Insights | Enabled | AI-generated analysis and suggestions |
| Executive Reports | Enabled | Markdown, JSON, and HTML reports |
| Per-Step Analysis | Enabled | Individual step health analysis |
| Yellow/Red Flags | Enabled | Warning and error detection per step |
| Rule-Based Recovery | Enabled | Works without LLM infrastructure |
| MCP Integration | Enabled | MCP tool registration available |

## Usage

### Command Line

```bash
# Full intelligent analysis
python src/gnn/24_intelligent_analysis.py --verbose

# Skip LLM analysis (rule-based only)
python src/gnn/24_intelligent_analysis.py --skip-llm

# Custom bottleneck threshold (seconds)
python src/gnn/24_intelligent_analysis.py --bottleneck-threshold 30.0

# Specific LLM model
python src/gnn/24_intelligent_analysis.py --analysis-model "gpt-4"
```

### Programmatic

```python
import logging
from pathlib import Path
from gnn.intelligent_analysis import process_intelligent_analysis

logger = logging.getLogger("pipeline")
result = process_intelligent_analysis(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output"),
    logger=logger,
)
```

## Output

Results are written to `output/24_intelligent_analysis_output/`:

- `intelligent_analysis_report.md` - Human-readable executive summary
- `analysis_data.json` - Machine-readable analysis data
- `intelligent_analysis_summary.json` - Compact summary with counts and paths

## Analysis Types

1. **Failure Analysis** - Root cause identification for failed steps
2. **Performance Analysis** - Bottleneck detection and timing analysis
3. **Health Scoring** - Pipeline and per-step health metrics
4. **Trend Analysis** - Pattern detection across runs
5. **Optimization Suggestions** - AI-powered improvement recommendations

## Dependencies

- **Required**: pathlib, json, logging, asyncio, datetime, dataclasses
- **Optional**: LLM processor (for AI-powered analysis)
- **Optional**: numpy, pandas (for statistical analysis)

## Data Flow

```
Pipeline Summary (00_pipeline_summary/)
    |
Intelligent Analysis
    |
    +-- Step Analysis (per-step flags)
    +-- Bottleneck Detection
    +-- Failure Root Cause
    +-- Executive Report (24_intelligent_analysis_output/)
```

---

## API Reference

### Data Classes

#### `StepAnalysis`

Detailed analysis result for a single pipeline step.

`step_number` preserves the summary's one-based execution order; use
`script_name` and the pipeline registry for the canonical step ID (0–24).

```python
class StepAnalysis:
    step_number: int  # One-based execution order in this run's summary
    script_name: str  # Script filename (e.g., "3_gnn.py")
    description: str  # Human-readable step description
    status: StepStatus  # pipeline.schemas step status ("SUCCESS", "FAILED", "WARNING", etc.)
    duration_seconds: float  # Execution time in seconds
    memory_mb: float  # Peak memory usage in MB
    exit_code: int  # Process exit code
    flags: List[str] = []  # List of flag description strings
    flag_type: str = "none"  # "none", "yellow", or "red"
    summary: str = ""  # One-line summary of step result
    stdout_snippet: str = ""  # Meaningful excerpt from stdout
    stderr_snippet: str = ""  # Meaningful excerpt from stderr
```

#### `AnalysisContext`

Context container for pipeline analysis, wrapping the raw summary data.

```python
@dataclass
class AnalysisContext:
    summary_data: Dict[str, Any]  # Raw pipeline execution summary
    timestamp: str  # ISO timestamp (auto-generated)
    pipeline_name: str = "GNN Pipeline"
    analysis_type: str = "comprehensive"
```

**Properties**:

- `overall_status -> str` - Pipeline status ("SUCCESS", "FAILED", "UNKNOWN")
- `total_duration -> float` - Total duration in seconds
- `steps -> List[Dict[str, Any]]` - List of step dictionaries
- `performance_summary -> Dict[str, Any]` - Performance summary data

**Methods**:

- `get_failed_steps() -> List[Dict[str, Any]]` - Filter to failed steps
- `get_successful_steps() -> List[Dict[str, Any]]` - Filter to successful steps
- `get_warning_steps() -> List[Dict[str, Any]]` - Filter to steps with warnings

### Processor Functions (`processor.py`)

#### `process_intelligent_analysis(target_dir, output_dir, logger, **kwargs) -> bool`

Main entry point. Orchestrates the full analysis pipeline: loads summary, analyzes steps, detects bottlenecks, extracts failures, generates recommendations, runs LLM analysis, and writes reports.

```python
def process_intelligent_analysis(
    target_dir: Path,
    output_dir: Path,
    logger: logging.Logger,
    **kwargs
) -> bool
```

**Parameters**:

- `target_dir` (Path): Input directory (not directly used, passed by pipeline convention)
- `output_dir` (Path): Output root directory. If the directory name is `24_intelligent_analysis_output`, outputs are written directly; otherwise a subdirectory is created.
- `logger` (logging.Logger): Logger instance for progress and error reporting
- `**kwargs`: Reserved for future options (skip_llm, bottleneck_threshold, etc.)

**Returns**: `bool` - `True` if analysis succeeded, `False` on critical failure (missing summary, JSON parse error, write failure)

**Example**:

```python
import logging
from pathlib import Path
from gnn.intelligent_analysis import process_intelligent_analysis

logger = logging.getLogger("pipeline")
success = process_intelligent_analysis(
    target_dir=Path("input/gnn_files"), output_dir=Path("output"), logger=logger
)
if not success:
    logger.error("Intelligent analysis failed")
```

#### `analyze_pipeline_summary(summary_data) -> Dict[str, Any]`

Extracts key metrics from the raw pipeline execution summary.

```python
def analyze_pipeline_summary(summary_data: Dict[str, Any]) -> Dict[str, Any]
```

**Parameters**:

- `summary_data` (Dict[str, Any]): Raw pipeline execution summary dictionary containing `steps`, `overall_status`, `total_duration_seconds`, and `performance_summary`

**Returns**: Dictionary with keys:

- `overall_status` (str): "SUCCESS", "FAILED", or "UNKNOWN"
- `total_duration` (float): Total pipeline duration in seconds
- `step_count` (int): Number of steps executed
- `failures` (List[Dict]): List of failure dicts with `step`, `error`, `duration`, `exit_code`
- `warnings` (List[Dict]): List of warning dicts with `step`, `message`
- `performance_metrics` (Dict): `peak_memory_mb`, `successful_steps`, `failed_steps`, `warning_count`
- `health_score` (float): 0-100 score based on success ratio and warning penalty

#### `analyze_individual_steps(summary_data) -> Tuple[List[StepAnalysis], Dict[str, List[StepAnalysis]]]`

Performs per-step analysis with flag detection based on duration, memory, exit codes, retries, and dependency warnings.

```python
def analyze_individual_steps(
    summary_data: Dict[str, Any]
) -> Tuple[List[StepAnalysis], Dict[str, List[StepAnalysis]]]
```

**Parameters**:

- `summary_data` (Dict[str, Any]): Pipeline execution summary

**Returns**: Tuple of:

1. `List[StepAnalysis]` - One analysis object per step
2. `Dict[str, List[StepAnalysis]]` - Steps grouped by flag type: `"red"`, `"yellow"`, `"green"`

**Flag Thresholds** (hardcoded):

- `SLOW_THRESHOLD`: 60.0 seconds (yellow flag)
- `VERY_SLOW_THRESHOLD`: 120.0 seconds (yellow flag)
- `HIGH_MEMORY_THRESHOLD`: 500.0 MB (yellow flag)
- `CRITICAL_MEMORY_THRESHOLD`: 1000.0 MB (yellow flag)
- Duration > 3x average (yellow flag)
- FAILED status or non-zero exit code (red flag)

#### `identify_bottlenecks(summary_data, threshold_seconds) -> List[Dict[str, Any]]`

Identifies steps that exceed a duration threshold or are significantly above average.

```python
def identify_bottlenecks(
    summary_data: Dict[str, Any],
    threshold_seconds: float = 60.0
) -> List[Dict[str, Any]]
```

**Parameters**:

- `summary_data` (Dict[str, Any]): Pipeline execution summary
- `threshold_seconds` (float): Duration threshold in seconds (default: 60.0)

**Returns**: List of bottleneck dicts sorted by duration descending, each with:

- `step` (str): Script name
- `duration_seconds` (float): Step duration
- `threshold_exceeded` (bool): Whether the absolute threshold was exceeded
- `above_average_ratio` (float): Ratio of step duration to average
- `memory_mb` (float): Peak memory usage

#### `extract_failure_context(summary_data) -> List[Dict[str, Any]]`

Extracts diagnostic context for failed steps, including preceding step information.

```python
def extract_failure_context(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]
```

**Returns**: List of failure context dicts, each with:

- `step_number`, `step_name`, `description`, `exit_code`
- `error_output` (str): Last 2000 chars of stderr
- `stdout_tail` (str): Last 1000 chars of stdout
- `duration`, `memory_at_failure`
- `preceding_step` (Dict or None): Name and status of previous step
- `dependency_warnings` (List[str])
- `prerequisite_check_passed` (bool)

#### `generate_recommendations(analysis, bottlenecks, flags_by_type) -> List[str]`

Generates prioritized, human-readable recommendations based on analysis results.

```python
def generate_recommendations(
    analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    flags_by_type: Dict[str, List]
) -> List[str]
```

**Returns**: List of recommendation strings with emoji prefixes for severity.

#### `generate_executive_report(...) -> str`

Generates a comprehensive markdown report with all analysis sections.

```python
def generate_executive_report(
    analysis: Dict[str, Any],
    bottlenecks: List[Dict[str, Any]],
    failures: List[Dict[str, Any]],
    recommendations: List[str],
    step_analyses: List[StepAnalysis],
    flags_by_type: Dict[str, List],
    llm_analysis: Optional[str] = None,
    summary_data: Optional[Dict[str, Any]] = None
) -> str
```

**Returns**: Complete markdown string with sections: header, quick overview table, AI analysis (or rule-based), red flags, yellow flags, per-step breakdown, detailed flagged step output, bottlenecks, recommendations, and pipeline configuration.

### Analyzer Functions (`analyzer.py`)

#### `IntelligentAnalyzer` Class

High-level analyzer with caching and LLM integration.

```python
class IntelligentAnalyzer:
    def __init__(
        self,
        context: Optional[AnalysisContext] = None,
        logger: Optional[logging.Logger] = None
    )
```

**Methods**:

- `set_context(context: AnalysisContext) -> None` - Set/replace context, clears cache
- `analyze() -> Dict[str, Any]` - Full analysis returning `timestamp`, `pipeline_name`, `overall_status`, `health_score`, `failure_analysis`, `performance_analysis`, `patterns`, `optimizations`
- `calculate_health_score() -> float` - Health score (0-100)
- `analyze_failures() -> Dict[str, Any]` - Failure details with severity distribution and common patterns
- `analyze_performance() -> Dict[str, Any]` - Duration/memory stats and bottlenecks
- `detect_patterns() -> List[Dict[str, Any]]` - Pattern detection (cascading failures, memory growth, etc.)
- `generate_optimizations() -> List[Dict[str, Any]]` - Optimization suggestions

**Example**:

```python
from gnn.intelligent_analysis.analyzer import IntelligentAnalyzer, AnalysisContext

context = AnalysisContext(summary_data=summary_json)
analyzer = IntelligentAnalyzer(context=context)
results = analyzer.analyze()
print(f"Health: {results['health_score']}/100")
print(f"Patterns: {len(results['patterns'])} detected")
```

#### `calculate_pipeline_health_score(summary_data) -> float`

Standalone health score calculation with weighted components.

```python
def calculate_pipeline_health_score(summary_data: Dict[str, Any]) -> float
```

**Scoring Formula** (total 100 points):

- Success rate: 40% weight (steps that did not fail)
- Warning rate: 20% weight (penalty for warnings)
- Duration efficiency: 20% weight (penalizes runs > 600 seconds)
- Memory efficiency: 20% weight (penalizes peak > 2048 MB)

#### `classify_failure_severity(step) -> str`

Classifies a failed step as `"critical"`, `"major"`, or `"minor"`.

```python
def classify_failure_severity(step: Dict[str, Any]) -> str
```

**Classification Rules**:

- **critical**: memory error, segfault, kernel died, fatal error, core dump, exit code > 127 or < 0
- **major**: exception, error:, failed to, cannot find, permission denied, timeout, not found
- **minor**: all other failures

#### `detect_performance_patterns(summary_data) -> List[Dict[str, Any]]`

Detects four pattern types.

```python
def detect_performance_patterns(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]
```

**Detected Patterns**:

1. `cascading_failure` - Consecutive failed steps (severity: high)
2. `memory_growth` - Memory increasing > 100MB/step (severity: medium)
3. `high_variance` - Duration variance > mean^2 (severity: low)
4. `late_failure` - Failures in the final 30% of pipeline (severity: medium)

#### `generate_optimization_suggestions(summary_data) -> List[Dict[str, Any]]`

Generates four categories of optimization suggestions.

```python
def generate_optimization_suggestions(summary_data: Dict[str, Any]) -> List[Dict[str, Any]]
```

**Suggestion Types**:

1. `parallelization` - Steps without dependency warnings (impact: high)
2. `caching` - Steps taking > 30 seconds (impact: medium)
3. `memory_optimization` - Peak memory > 1024 MB (impact: medium)
4. `early_termination` - Failed steps consuming > 30% of total duration (impact: high)

### Module-Level Utility Functions (`__init__.py`)

#### `get_module_info() -> Dict[str, Any]`

Returns version, description, features list, report formats, and LLM backends.

#### `get_supported_analysis_types() -> List[str]`

Returns list of supported analysis type strings.

#### `validate_pipeline_summary(summary) -> bool`

Checks that summary dict contains required fields: `start_time`, `steps`, `overall_status`.

#### `check_analysis_tools() -> Dict[str, Dict[str, Any]]`

Checks availability of `llm_processor`, `numpy`, and `pandas`. Returns availability status and versions.

---

## Error Handling

### Graceful Degradation Strategy

The module implements a layered recovery approach:

| Scenario | Primary Path | Recovery | Behavior |
|----------|-------------|----------|----------|
| LLM available | `_run_llm_analysis()` via `llm.llm_processor` | `_generate_rule_based_summary()` | Full AI-powered executive summary |
| LLM import fails | Catches `ImportError` on `llm.llm_processor` | Rule-based summary | Structured summary without AI |
| LLM API call fails | Catches generic `Exception` on `processor.get_response()` | Rule-based summary | Falls back with error logged |
| LLM processor returns None | Checks `if not processor` | Rule-based summary | Immediate recovery |
| Pipeline summary missing | Checks `summary_path.exists()` | Returns `False` | Logs error, step marked failed |
| JSON parse failure | Catches `Exception` on `json.load()` | Returns `False` | Logs error with exception details |
| Report write failure | Catches `Exception` on file write | Returns `False` | Logs error, analysis data may still save |
| Analysis data write failure | Catches `Exception` on JSON dump | Continues | Logs warning, report file still valid |
| numpy/pandas missing | Not required at runtime | Pure Python analysis | All core functions work without them |

### Error Flow in `process_intelligent_analysis()`

```
1. Load summary JSON
   -> Missing file: log error, return False
   -> Parse error: log error, return False

2. Analyze pipeline summary
   -> Always succeeds (handles missing keys with .get())

3. Analyze individual steps
   -> Always succeeds (graceful handling of missing data)

4. Identify bottlenecks
   -> Always succeeds (returns empty list if no data)

5. Extract failure context
   -> Always succeeds (returns empty list if no failures)

6. Generate recommendations
   -> Always succeeds (returns list based on available data)

7. Run LLM analysis
   -> Import error: recovery to rule-based
   -> API error: recovery to rule-based
   -> Timeout: recovery to rule-based

8. Generate executive report
   -> Always succeeds (string concatenation)

9. Save outputs
   -> Report write failure: return False
   -> Data write failure: log warning, continue
   -> Summary write failure: log warning, continue
```

---

## Testing Guide

### Test Files

- `tests/intelligent_analysis/test_intelligent_analysis_overall.py` - Module-level tests
- Test via full pipeline: `python src/gnn/main.py --only-steps 24 --verbose`

### Required Test Fixtures

```python
import pytest
from pathlib import Path
from gnn.intelligent_analysis.processor import (
    analyze_pipeline_summary,
    analyze_individual_steps,
    identify_bottlenecks,
    extract_failure_context,
    generate_recommendations,
    StepAnalysis,
)
from gnn.intelligent_analysis.analyzer import (
    IntelligentAnalyzer,
    AnalysisContext,
    calculate_pipeline_health_score,
    classify_failure_severity,
    detect_performance_patterns,
    generate_optimization_suggestions,
)


@pytest.fixture
def sample_summary_data():
    """Minimal pipeline summary for testing."""
    return {
        "overall_status": "SUCCESS",
        "total_duration_seconds": 120.0,
        "performance_summary": {
            "peak_memory_mb": 256.0,
            "successful_steps": 5,
            "failed_steps": 0,
            "warnings": 0,
        },
        "steps": [
            {
                "step_number": 3,
                "script_name": "3_gnn.py",
                "description": "GNN parsing",
                "status": "SUCCESS",
                "duration_seconds": 2.5,
                "peak_memory_mb": 50.0,
                "exit_code": 0,
                "stdout": "Parsed 1 model",
                "stderr": "",
            },
            {
                "step_number": 12,
                "script_name": "12_execute.py",
                "description": "Execute simulations",
                "status": "FAILED",
                "duration_seconds": 45.0,
                "peak_memory_mb": 200.0,
                "exit_code": 1,
                "stdout": "",
                "stderr": "ModuleNotFoundError: No module named 'pymdp'",
            },
        ],
    }


@pytest.fixture
def failed_summary_data():
    """Pipeline summary with multiple failures for testing."""
    return {
        "overall_status": "FAILED",
        "total_duration_seconds": 300.0,
        "performance_summary": {
            "peak_memory_mb": 1500.0,
            "successful_steps": 2,
            "failed_steps": 3,
            "warnings": 2,
        },
        "steps": [
            {
                "step_number": i,
                "script_name": f"{i}_step.py",
                "description": f"Step {i}",
                "status": "FAILED" if i in [10, 11, 12] else "SUCCESS",
                "duration_seconds": 60.0 * (i + 1),
                "peak_memory_mb": 100.0 * (i + 1),
                "exit_code": 1 if i in [10, 11, 12] else 0,
                "stderr": "MemoryError" if i == 10 else "",
            }
            for i in range(5)
        ],
    }
```

### Example Test Patterns

```python
def test_analyze_pipeline_summary_success(sample_summary_data):
    """Test analysis of a successful pipeline run."""
    result = analyze_pipeline_summary(sample_summary_data)
    assert result["overall_status"] == "SUCCESS"
    assert result["step_count"] == 2
    assert result["health_score"] > 0
    assert len(result["failures"]) == 1  # One failed step
    assert result["failures"][0]["step"] == "12_execute.py"


def test_analyze_individual_steps_flags(sample_summary_data):
    """Test that flags are correctly assigned to steps."""
    step_analyses, flags_by_type = analyze_individual_steps(sample_summary_data)
    assert len(step_analyses) == 2
    # The failed step should have a red flag
    failed = [s for s in step_analyses if s.flag_type == "red"]
    assert len(failed) == 1
    assert failed[0].script_name == "12_execute.py"


def test_identify_bottlenecks_threshold(sample_summary_data):
    """Test bottleneck detection with custom threshold."""
    bottlenecks = identify_bottlenecks(sample_summary_data, threshold_seconds=10.0)
    assert len(bottlenecks) >= 1
    assert bottlenecks[0]["step"] == "12_execute.py"


def test_health_score_perfect():
    """Test health score for a perfect pipeline run."""
    perfect = {
        "steps": [
            {"status": "SUCCESS", "duration_seconds": 1.0, "peak_memory_mb": 10.0}
        ],
        "total_duration_seconds": 1.0,
        "performance_summary": {"peak_memory_mb": 10.0},
    }
    score = calculate_pipeline_health_score(perfect)
    assert score > 90.0


def test_classify_failure_severity_critical():
    """Test that memory errors are classified as critical."""
    step = {"stderr": "MemoryError: cannot allocate", "exit_code": 137}
    assert classify_failure_severity(step) == "critical"


def test_classify_failure_severity_major():
    """Test that import errors are classified as major."""
    step = {"stderr": "ImportError: No module named 'foo'", "exit_code": 1}
    assert classify_failure_severity(step) == "major"


def test_detect_cascading_failure_pattern():
    """Test detection of consecutive failures."""
    data = {
        "steps": [
            {"status": "SUCCESS", "duration_seconds": 1.0, "peak_memory_mb": 10.0},
            {"status": "FAILED", "duration_seconds": 1.0, "peak_memory_mb": 10.0},
            {"status": "FAILED", "duration_seconds": 1.0, "peak_memory_mb": 10.0},
        ],
        "performance_summary": {},
    }
    patterns = detect_performance_patterns(data)
    assert any(p["type"] == "cascading_failure" for p in patterns)
```

### LLM Unavailable Handling

```python
def test_process_without_provider_config(sample_summary_data, tmp_path):
    """Test that provider unavailability still yields rule-based analysis."""
    import logging

    logger = logging.getLogger("test")

    # Write summary to tmp_path
    summary_dir = tmp_path / "00_pipeline_summary"
    summary_dir.mkdir()
    import json

    (summary_dir / "pipeline_execution_summary.json").write_text(
        json.dumps(sample_summary_data)
    )

    from gnn.intelligent_analysis.processor import process_intelligent_analysis

    result = process_intelligent_analysis(
        target_dir=tmp_path, output_dir=tmp_path, logger=logger
    )
    assert result is True
```

---

## Configuration

### Configurable Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bottleneck_threshold` | `float` | `60.0` | Duration threshold in seconds for flagging slow steps via `identify_bottlenecks()` |
| `analysis_model` | `str` | `None` → `OLLAMA_MODEL` → `llm.defaults.DEFAULT_OLLAMA_MODEL` | LLM model tag for `_run_llm_analysis()` |
| `skip_llm` | `bool` | `False` | When True, skips LLM analysis and uses rule-based recovery only |
| `max_tokens` | `int` | `2500` | Maximum tokens for LLM response in `_run_llm_analysis()` |

### Hardcoded Thresholds in `analyze_individual_steps()`

| Threshold | Value | Flag Level | Description |
|-----------|-------|------------|-------------|
| `SLOW_THRESHOLD` | `60.0`s | Yellow | Step exceeds 1 minute |
| `VERY_SLOW_THRESHOLD` | `120.0`s | Yellow | Step exceeds 2 minutes |
| `HIGH_MEMORY_THRESHOLD` | `500.0` MB | Yellow | Step memory exceeds 500 MB |
| `CRITICAL_MEMORY_THRESHOLD` | `1000.0` MB | Yellow | Step memory exceeds 1 GB |
| Duration > 3x average | Dynamic | Yellow | Step is significantly above average |

### Health Score Weights in `calculate_pipeline_health_score()`

| Component | Weight | Baseline | Description |
|-----------|--------|----------|-------------|
| Success rate | 40% | All steps pass | Ratio of non-failed steps |
| Warning rate | 20% | No warnings | Penalty for warning steps |
| Duration efficiency | 20% | < 600 seconds | Penalizes runs over 10 minutes |
| Memory efficiency | 20% | < 2048 MB | Penalizes peak memory over 2 GB |

### Recommendation Thresholds in `generate_recommendations()`

| Metric | Threshold | Recommendation |
|--------|-----------|----------------|
| Peak memory | > 2048 MB | Memory optimization recommended |
| Peak memory | > 1024 MB | Memory monitoring recommended |
| Health score | < 70 | "Needs attention" |
| Health score | < 70 | "Critical - address failures" |
| Red flags | > 0 | "Immediate attention required" |

---

## Integration Points

### Pipeline Position

Step 24 is the **final step** in the GNN pipeline. It runs after all other steps have completed and analyzes the full execution results.

### Input Data

| Source | File | Description |
|--------|------|-------------|
| Pipeline Orchestrator | `output/00_pipeline_summary/pipeline_execution_summary.json` | Primary input: complete execution summary with per-step data |

The module looks for the summary file at two locations:

1. `{output_dir}/00_pipeline_summary/pipeline_execution_summary.json`
2. `{output_dir}/../00_pipeline_summary/pipeline_execution_summary.json` (recovery)

### Output Data

| File | Format | Consumers |
|------|--------|-----------|
| `intelligent_analysis_report.md` | Markdown | Human readers, report step (23) |
| `analysis_data.json` | JSON | Programmatic consumers, MCP tools |
| `intelligent_analysis_summary.json` | JSON | Pipeline dashboard, monitoring |

### Dependency Chain

```
Steps 0-24 (all pipeline steps)
    |
    v
00_pipeline_summary/pipeline_execution_summary.json
    |
    v
Step 24: intelligent_analysis
    |
    +-- Reads: pipeline_execution_summary.json
    +-- Optionally uses: llm.llm_processor (for AI analysis)
    +-- Writes: 24_intelligent_analysis_output/
```

### Steps That Must Complete Before Step 24

- Step 0 (Template): Provides pipeline framework
- Any steps that were actually executed (their results appear in the summary)
- The pipeline orchestrator (`main.py`) must have written the execution summary

### Steps That Depend on Step 24

- None (Step 24 is the terminal step)
- However, the report step (23) may reference analysis outputs if re-run

---

## Performance

### Expected Execution Times

| Mode | Duration | Notes |
|------|----------|-------|
| Rule-based only (`--skip-llm`) | 0.5 - 2 seconds | Pure Python, no external calls |
| With LLM (Ollama, default smollm2 instruct) | 3 - 10 seconds | Depends on model and prompt size |
| With LLM (cloud API) | 5 - 30 seconds | Network latency + processing |

### Memory Usage

- **Base analysis**: ~20-30 MB (JSON parsing + string processing)
- **With numpy/pandas**: ~50-80 MB additional if those libraries are imported
- **LLM processor**: Depends on provider (Ollama can use 200MB-6GB for the model itself)

### LLM API Cost Considerations

- The module sends a single prompt per pipeline run (not per step)
- Prompt size scales with step count (approximately 100-500 tokens per step in the summary)
- Response is limited to `max_tokens=2500`
- Using local Ollama incurs no API cost
- Cloud API cost is typically < $0.01 per analysis run with GPT-3.5 or similar

### Optimization Tips

1. Use `--skip-llm` for fast iteration during development
2. The rule-based recovery produces actionable results without any LLM
3. If using Ollama, prefer the default small instruct tag or set `OLLAMA_MODEL` to a faster local pull
4. The analysis itself (excluding LLM) is CPU-bound and single-threaded

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|---------|
| "Pipeline summary not found" | Step 24 ran before the pipeline orchestrator wrote the summary, or wrong output directory | Ensure `main.py` completed or manually verify `output/00_pipeline_summary/pipeline_execution_summary.json` exists |
| "Failed to load pipeline summary" | Corrupted or malformed JSON in the summary file | Re-run the pipeline to regenerate the summary; check for partial writes or disk space issues |
| LLM analysis returns rule-based output | Ollama not running, no API keys configured, or LLM import error | Start Ollama (`ollama serve`), verify models installed (`ollama list`), or set API keys |
| "Error calling LLM" in logs | LLM API timeout, rate limit, or network error | Check Ollama service status; increase timeout; verify network connectivity |
| Health score is 0.0 | Empty steps list in summary data | Verify pipeline actually executed steps; check that `steps` array is populated |
| All steps show "green" despite failures | Summary data has incorrect `status` field values | Check that pipeline orchestrator correctly records step exit codes and status |
| Report file is empty or truncated | Disk full or write permission error | Check disk space and directory permissions for `output/24_intelligent_analysis_output/` |
| `asyncio.run()` error | Already running in an async context (e.g., Jupyter notebook) | Use `await _run_llm_analysis()` directly or run from a synchronous context |

---

## Architecture Decisions

### Why Rule-Based Recovery Alongside LLM?

The module was designed to always produce useful output, regardless of LLM availability. In production environments:

1. **Ollama may not be installed** - Many CI/CD environments lack local LLM infrastructure
2. **API keys may not be configured** - Not all deployments have cloud LLM access
3. **LLM responses are non-deterministic** - Rule-based output provides consistent baselines
4. **Speed matters** - Rule-based analysis completes in milliseconds vs seconds for LLM
5. **Cost control** - Repeated pipeline runs should not require API spending

The `_generate_rule_based_summary()` function mirrors the LLM prompt's expected output format (Executive Summary, Red Flags, Yellow Flags, Action Items) so that the executive report structure remains consistent regardless of which analysis path was used.

### Why Separate processor.py and analyzer.py?

The split follows the project's thin orchestrator pattern:

- **`processor.py`**: Contains the pipeline-facing functions (`process_intelligent_analysis`, `generate_executive_report`, etc.) that handle I/O, file operations, and orchestration logic. This is the "glue" between the pipeline framework and the analysis logic.

- **`analyzer.py`**: Contains pure analysis logic (`IntelligentAnalyzer`, `calculate_pipeline_health_score`, `classify_failure_severity`, etc.) with no file I/O or pipeline awareness. These functions operate on data structures and can be tested in isolation.

This separation means:

- `analyzer.py` can be imported and used independently of the pipeline
- `processor.py` handles all side effects (file reads/writes, logging, LLM calls)
- Testing is simpler: analyzer tests need no filesystem fixtures

### Why asyncio for LLM Calls?

The `_run_llm_analysis()` function is async because it uses the shared `llm.llm_processor` infrastructure, which is built on async patterns for concurrent provider support. The `process_intelligent_analysis()` entry point bridges this with `asyncio.run()` for synchronous callers.

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:

- Per-step analysis with yellow/red flag detection
- Rule-based recovery analysis
- LLM-powered executive summaries
- Health scoring with weighted components
- Pattern detection (cascading failures, memory growth, etc.)
- Optimization suggestions
- Comprehensive executive report generation

### Roadmap

- **Next**: Historical trend analysis across multiple pipeline runs
- **Future**: Real-time monitoring integration, anomaly detection

---

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Analysis Module](../analysis/AGENTS.md)
- [LLM Module](../llm/AGENTS.md)
- [Execute Module](../execute/AGENTS.md)

### Internal References

- `src/gnn/main.py` - Pipeline orchestrator that generates the summary consumed by this module
- `src/gnn/llm/llm_processor.py` - LLM processor used for AI-powered analysis
- `src/gnn/utils/pipeline_orchestration/pipeline_template.py` - Logging utilities used throughout

---

**Last Updated**: 2026-09-24
**Maintainer**: GNN Pipeline Team
**Status**: Production Ready
**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: Thin Orchestrator Pattern

---

## Documentation

- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
