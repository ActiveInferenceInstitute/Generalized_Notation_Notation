# Step 17: Integration

## Architectural Mapping

**Orchestrator**: `src/gnn/17_integration.py` (52 lines)
**Implementation Layer**: `src/gnn/integration/`

## Module Description

This module provides comprehensive system integration capabilities for the GNN pipeline, enabling cross-module coordination, data flow management, and system-wide functionality integration.


```
src/gnn/integration/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── processor.py                   # process_integration: dependency-graph coordination and consistency checks
├── meta_analysis/                 # Parameter-sweep runtime/simulation meta-analysis (collector, statistics, validator, visualizer)
└── mcp.py                         # Model Context Protocol integration
```



Main function for processing system integration tasks.

## Agent Identity & Capabilities

# Integration Module - Agent Scaffolding

## Module Overview

**Purpose**: System integration and consistency validation using Graph Theory (NetworkX) and cross-module reference checking.

**Pipeline Step**: Step 17: Integration (17_integration.py)

**Category**: System Integration / Coordination

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities
1. Coordinate cross-module interactions and data flow
2. Provide recovery implementations for missing dependencies
3. Manage system-wide configuration and state
4. Enable seamless integration between pipeline steps
5. Handle inter-module communication and data exchange

### Key Capabilities
- **Dependency Graph Construction**: Uses `networkx` to build a directed graph of system components.
- **Cycle Detection**: Identifies circular dependencies that could cause infinite loops or initialization errors.
- **Cross-Reference Validation**: Ensures all referenced components are defined using explicit filename checks.
- **System Stats**: Reports node/edge counts and graph density.

---

## API Reference

### Public Functions

#### `process_integration(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main integration processing function called by orchestrator (17_integration.py). Coordinates cross-module interactions and validates system consistency using graph theory.

**Parameters**:
- `target_dir` (Path): Directory containing pipeline outputs to integrate
- `output_dir` (Path): Output directory for integration results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `**kwargs`: Accepted but not read by the current `process_integration` implementation — no `integration_mode`, `system_coordination`, `validate_dependencies`, or `detect_cycles` key has any effect.

**Returns**: `bool` - True if integration processing succeeded, False otherwise

**Example**:
```python
from integration import process_integration
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
success = process_integration(
    target_dir=Path("output"),
    output_dir=Path("output/17_integration_output"),
    logger=logger,
    verbose=True,
    integration_mode="coordinated",
    validate_dependencies=True,
)
```

**Note**: Dependency-graph coordination is performed internally by `process_integration` itself (there is no separate `coordinate_pipeline_modules()` function). It builds a `networkx` directed graph of components, detects cycles and isolated nodes, and stores the results under `results["system_graph_stats"]` with keys `nodes`, `edges`, `cycles`, `isolated_nodes`, and `components`.

---

## Dependencies

### Required Dependencies
- `pathlib` - Path manipulation and file system operations
- `typing` - Type hints and annotations
- `logging` - Logging and progress reporting

### Optional Dependencies
- `psutil` - System resource monitoring (recovery: basic monitoring)
- `requests` - HTTP communication (recovery: local only)

### Internal Dependencies
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized pipeline processing patterns
- `pipeline.config` - Pipeline configuration management

---

## Configuration

No environment variables or config files are read by `src/gnn/integration/`; behavior is controlled by the `process_integration(...)` parameters documented above.

## Usage Examples

### Basic Usage
```python
from integration.processor import process_integration

success = process_integration(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/17_integration_output"),
    logger=logger,
    integration_mode="coordinated",
)
```

---

## Output Specification

### Output Products
- `integration_processing_summary.json` - Integration processing summary
- `system_coordination_report.json` - Cross-module coordination status
- `integration_status.json` - Current integration state

### Output Directory Structure
```
output/17_integration_output/
├── integration_processing_summary.json
├── system_coordination_report.json
└── integration_status.json
```

---

## Performance Characteristics

### Performance
- **Fast Path**: <1s for basic graph validation
- **Analysis Depth**: O(N+E) complexity for cycle detection
- **Memory**: Proportional to graph size (Node/Edge count)

---

## Error Handling

### Graceful Degradation
- **No external dependencies**: Local-only integration mode
- **Module unavailable**: Skip integration for that module
- **Network issues**: Recovery to local coordination only

### Error Categories
1. **Coordination Errors**: Unable to coordinate between modules
2. **Dependency Errors**: Missing required integration dependencies
3. **Configuration Errors**: Invalid integration settings

---

## Integration Points

### Orchestrated By
- **Script**: `17_integration.py` (Step 17)
- **Function**: `process_integration()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `tests/integration/test_integration_overall.py` - Module-level integration tests
- `main.py` - Pipeline orchestration

### Data Flow
```
Pipeline Steps → Integration Coordination → System State → Cross-Module Communication → Unified Output
```

---

## Testing

### Test Files
- `tests/integration/test_integration_functional.py` - Functional integration tests
- `tests/integration/test_integration_processor.py` - Processor-level integration tests

### Test Coverage
- Measure: `uv run --extra dev python -m pytest tests/integration/ --cov=gnn.integration --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios
1. Cross-module coordination with various step combinations
2. Recovery mode operation when dependencies unavailable
3. System state synchronization accuracy
4. Error handling with partial module failures

---

## MCP Integration

### Tools Registered

Registered in `register_tools` (`src/gnn/integration/mcp.py`):

- `process_integration` - Run Step 17 over pipeline outputs
- `list_supported_integrations` - List supported integration checks
- `get_integration_status` - Read current integration status
- `check_integration_dependencies` - Report integration dependencies

### MCP File Location
- `src/gnn/integration/mcp.py` - MCP tool registrations

---

## Troubleshooting

### Common Issues

#### Issue 1: Circular dependency detection fails
**Symptom**: Cycle detection reports false positives or misses cycles  
**Cause**: Dependency graph construction incomplete or incorrect  
**Solution**: 
- Verify all modules are properly discovered
- Check module import statements are correct
- Use `--verbose` flag for detailed dependency graph
- Review dependency graph visualization

#### Issue 2: Cross-reference validation errors
**Symptom**: Valid references reported as missing  
**Cause**: File path resolution issues or incorrect reference format  
**Solution**:
- Verify file paths are relative to project root
- Check reference format matches expected pattern
- Ensure referenced files exist in expected locations
- Review cross-reference validation logs

---

## Version History

### Current module status

**Features**:
- Cross-module coordination
- Dependency graph construction
- Cycle detection
- Cross-reference validation
- System-wide configuration

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced dependency analysis
- **Future**: Real-time integration monitoring

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Pipeline Configuration](../../../src/gnn/integration/../pipeline/AGENTS.md)

### External Resources
- [NetworkX Documentation](https://networkx.org/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/integration/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/integration/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/integration/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/integration/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/integration](../../../src/gnn/integration)
