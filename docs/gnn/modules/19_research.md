# Step 19: Research

## Architectural Mapping

**Orchestrator**: `src/gnn/19_research.py` (52 lines)
**Implementation Layer**: `src/gnn/research/`

## Module Description

This module provides comprehensive research tools and experimental features for Active Inference research, including advanced analysis, experimental workflows, and research methodology support.


```
src/gnn/research/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── processor.py                   # process_research + rule-based hypothesis-generation helpers
└── mcp.py                         # Model Context Protocol integration
```



Main function for processing research-related tasks.

## Agent Identity & Capabilities

# Research Module - Agent Scaffolding

## Module Overview

**Purpose**: Rule-based expert system for analyzing GNN models and generating research hypotheses.

**Pipeline Step**: Step 19: Research tools (19_research.py)

**Category**: Research / Experimental Analysis

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Advanced research analysis and experimentation
2. Research methodology implementation and validation
3. Experimental feature development and testing
4. Research data collection and analysis
5. Publication and documentation support
6. Research collaboration tools

### Key Capabilities
- **Rule-Based Hypothesis Generation**: Uses static analysis heuristics to suggest model improvements.
- **Complexity Analysis**: Detects high-dimensional matrices (>10 dims) to suggest reduction techniques.
- **Structural Diagnostics**: Analyzes variable-to-connection ratios to identify sparse causal structures.
- **Automated Reporting**: Generates markdown reports justifying every hypothesis with discovered evidence.

---

## API Reference

### Public Functions

#### `process_research(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main research processing function called by orchestrator (19_research.py). Performs rule-based hypothesis generation and research analysis.

**Parameters**:
- `target_dir` (Path): Directory containing research data (GNN files)
- `output_dir` (Path): Output directory for research results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance (default: None)
- `analysis_type` (str, optional): Type of analysis ("comprehensive", "statistical", "experimental") (default: "comprehensive")
- `generate_hypotheses` (bool, optional): Generate research hypotheses (default: True)
- `**kwargs`: Additional research options

**Returns**: `bool` - True if research processing succeeded, False otherwise

**Example**:
```python
from research import process_research
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_research(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/19_research_output"),
    logger=logger,
    verbose=True,
    analysis_type="comprehensive",
)
```

#### `generate_rule_based_hypotheses(content: str, model_family: str, dims: Dict[str, List[int]], connections: Dict[str, int]) -> List[Dict[str, Any]]`
**Description**: Generate research hypotheses via rule-based static analysis (dimensionality-reduction and sparse-connectivity diagnostics, among other domain-specific rules for Active Inference / generative model research).

**Parameters**:
- `content` (str): Raw GNN file content
- `model_family` (str): Detected model family (see `detect_model_family`)
- `dims` (Dict[str, List[int]]): Extracted state-space dimensions (see `extract_state_space_dims`)
- `connections` (Dict[str, int]): Connection counts (see `count_connections`)

**Returns**: `List[Dict[str, Any]]` - Hypotheses, each with `type`, `description`, `rationale`, and `priority` (`"high"`/`"medium"`/`"low"`)

#### `detect_model_family(content: str) -> str`
**Description**: Detect the Active Inference model family from raw GNN file content.

#### `extract_state_space_dims(content: str) -> Dict[str, List[int]]`
**Description**: Extract structured state-space variable dimensions from GNN file content.

#### `count_connections(content: str) -> Dict[str, int]`
**Description**: Count connections in GNN file content, used to compute variable-to-connection ratios for sparsity diagnostics.

**Note**: There is no separate report-generation function — `process_research` itself writes both `research_results.json` and a markdown `research_report.md` (grouped by hypothesis priority) directly to `output_dir`.

---

## Dependencies

### Required Dependencies
- `numpy` - Numerical computations
- `pandas` - Data analysis
- `matplotlib` - Research visualization

### Optional Dependencies
- `scipy` - Advanced statistical analysis
- `scikit-learn` - Machine learning research tools
- `jupyter` - Interactive research notebooks

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

---

## Configuration

### Research Settings
```python
RESEARCH_CONFIG = {
    "analysis_types": ["statistical", "experimental", "comparative"],
    "output_formats": ["markdown", "html", "pdf"],
    "visualization_style": "publication",
    "statistical_significance": 0.05,
    "include_methodology": True,
}
```

---

## Usage Examples

### Basic Research Analysis
```python
from research.processor import process_research

success = process_research(
    target_dir="research_data/",
    output_dir="output/19_research_output",
    analysis_type="comprehensive",
)
```

### Rule-Based Hypothesis Generation
```python
from pathlib import Path
from research.processor import (
    detect_model_family,
    extract_state_space_dims,
    count_connections,
    generate_rule_based_hypotheses,
)

content = Path("models/my_model.md").read_text()
model_family = detect_model_family(content)
dims = extract_state_space_dims(content)
connections = count_connections(content)

hypotheses = generate_rule_based_hypotheses(content, model_family, dims, connections)
for h in hypotheses:
    print(f"[{h['priority']}] {h['type']}: {h['description']}")
```

Report generation is not a separate call — `process_research` writes `research_results.json` and `research_report.md` automatically as part of processing.

---

## Output Specification

### Output Products
- `research_analysis_report.md` - Comprehensive research report
- `research_data_analysis.json` - Detailed analysis results
- `research_visualizations/` - Research visualizations
- `research_summary.json` - Research summary

### Output Directory Structure
```
output/19_research_output/
├── research_analysis_report.md
├── research_data_analysis.json
├── research_visualizations/
│   ├── statistical_plots.png
│   └── experimental_results.png
└── research_summary.json
```

---

## Performance Characteristics

### Latest Execution
- **Duration**: Variable (depends on research complexity)
- **Memory**: ~50-200MB for complex analyses
- **Status**: ✅ Production Ready

### Expected Performance
- **Statistical Analysis**: 1-5 minutes
- **Experimental Analysis**: 5-30 minutes
- **Report Generation**: < 1 minute
- **Visualization**: 30 seconds - 2 minutes

---

## Error Handling

### Research Errors
1. **Data Quality Issues**: Invalid or insufficient research data
2. **Analysis Failures**: Statistical or computational errors
3. **Visualization Errors**: Plot generation failures
4. **Report Generation**: Documentation creation errors

### Recovery Strategies
- **Data Cleaning**: Automatic data quality improvement
- **Analysis Recovery**: Alternative analysis methods
- **Visualization Recovery**: Simplified visualizations
- **Report Recovery**: Error-aware report generation

---

## Integration Points

### Orchestrated By
- **Script**: `19_research.py` (Step 19)
- **Function**: `process_research()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Pipeline utilities

### Imported By
- Research-specific applications
- `tests.test_research_*` - Research tests

### Data Flow
```
Research Data → Analysis → Visualization → Report Generation → Publication
```

---

## Testing

### Test Files
- `tests/research/test_research_overall.py` - Module-level tests
- `tests/research/test_research_functional.py` - Functional tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/research/ --cov=gnn.research --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Research analysis with various data types
2. Report generation and formatting
3. Visualization creation
4. Error handling and recovery

---

## MCP Integration

### Tools Registered

Registered in `register_tools` (`src/gnn/research/mcp.py`):

- `process_research` - Run Step 19 over a directory
- `list_research_topics` - List research topics and domains
- `read_research_results` - Read saved research results
- `get_research_module_info` - Module metadata

### MCP File Location
- `src/gnn/research/mcp.py` - MCP tool registrations


---

## Troubleshooting

### Common Issues

#### Issue 1: Hypothesis generation produces no results
**Symptom**: Research analysis completes but no hypotheses generated  
**Cause**: Model structure doesn't match rule patterns or analysis incomplete  
**Solution**: 
- Verify GNN model has complete structure
- Check that model has variables and connections
- Use `--verbose` flag for detailed analysis logs
- Review rule-based analysis patterns

#### Issue 2: Research report generation fails
**Symptom**: Analysis succeeds but report generation errors  
**Cause**: Report template issues or output format problems  
**Solution**:
- Check output directory permissions
- Verify report format is supported
- Review report template structure
- Use default markdown format if issues persist

---

## Version History

### Current module status

**Features**:
- Rule-based hypothesis generation
- Complexity analysis
- Structural diagnostics
- Automated reporting

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced hypothesis generation
- **Future**: Machine learning-based hypothesis generation

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Research Module](../../../src/gnn/research/../research/README.md)

### External Resources
- [Active Inference Research](../../research/README.md)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern

---
## Documentation
- **[README](../../../src/gnn/research/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/research/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/research/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/research/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/research](../../../src/gnn/research)
