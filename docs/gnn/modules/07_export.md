# Step 7: Export

## Architectural Mapping

**Orchestrator**: `src/gnn/7_export.py` (128 lines)
**Implementation Layer**: `src/gnn/export/`

## Module Description

This module provides comprehensive multi-format export capabilities for GNN models, supporting JSON, XML, GraphML, GEXF, Pickle, and other formats with semantic preservation and cross-format compatibility.


```
src/gnn/export/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── AGENTS.md                      # Agent scaffolding documentation
├── SPEC.md                        # Module specification
├── core.py                        # Pipeline integration adapter
├── processor.py                   # Export orchestration and GNN parsing
├── formatters.py                  # Format-specific serializers
├── format_exporters.py            # Advanced GNN-aware exporters

## Agent Identity & Capabilities

# Export Module - Agent Scaffolding

## Module Overview

**Purpose**: Multi-format export generation (JSON, XML, GraphML, GEXF, Pickle) from parsed GNN models

**Pipeline Step**: Step 7: Multi-format export (7_export.py)

**Category**: Data Export / Transformation

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities

1. Export parsed GNN models to multiple formats
2. Generate graph-based representations (GraphML, GEXF)
3. Create portable serializations (JSON, XML, Pickle)
4. Validate export integrity
5. Provide format-specific documentation

### Key Capabilities

- JSON export with schema validation
- XML export with DTD/XSD
- GraphML for network analysis tools
- GEXF for Gephi visualization
- Pickle for Python persistence

---

## API Reference

### Public Functions

#### `process_export(target_dir, output_dir, verbose=False, **kwargs) -> bool`

**Description**: Main export processing function (`src/gnn/export/processor.py`), invoked by `src/gnn/7_export.py`. Loads the parsed GNN results written by Step 3 (`gnn_processing_results.json`) from the base output directory and exports each parsed model to multiple formats.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for exports
- `verbose` (bool): Enable verbose logging (default: False)
- `**kwargs`: Additional processing options, including `formats` and an
  optional opt-in `geo_infer` mapping that enables the strict GEO_INFER
  export (see Configuration).

**Returns**: `True` if all exports succeeded

**Example**:

```python
from export import process_export

success = process_export(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    verbose=True
)
```

---

## Supported Export Formats

### Standard Formats

1. **JSON**: Human-readable, widely compatible
2. **XML**: Schema-validated, industry standard

### Graph Formats

3. **GraphML**: Standard graph format (Cytoscape, yEd)
2. **GEXF**: Gephi visualization format

### Text Formats

5. **Plaintext Summary**: Human-readable model overview
2. **Plaintext DSL**: Round-trip GNN-like text

### Binary Formats

7. **Pickle**: Fast Python serialization

---

## Configuration

### Configuration Options

#### Export Format Selection

- `export_formats` (List[str]): Formats to export (default: `["json", "xml", "graphml", "gexf", "pickle"]`)
  - `"json"`: JSON export (human-readable)
  - `"xml"`: XML export (schema-validated)
  - `"graphml"`: GraphML format (Cytoscape, yEd)
  - `"gexf"`: GEXF format (Gephi)
  - `"pickle"`: Python pickle serialization
  - `"txt"`: Plaintext summary
  - `"dsl"`: Plaintext DSL

Passing `geo_infer={"step_seconds": ..., "state_ids_path": ..., "space_kind": ...}`
to `process_export` (or passing the Step 7 CLI flags `--geo-step-seconds`,
`--geo-state-ids`, `--geo-space-kind`, registered in the shared step-argument
registry) requests the strict GEO_INFER artifact
(`{model}_geo_infer.geo-infer.json`) in addition to the requested formats.
Rules:

- `step_seconds` (float): mandatory, finite, and positive.
- `state_ids_path` (str): optional path to a JSON array labeling states in
  matrix order; required when `space_kind` is `"h3"`.
- `space_kind` (str): `"categorical"` (default) or `"h3"`.
- **Visible failure**: requesting `geo_infer` without `step_seconds` raises
  a distinct error naming `geo_infer` and the missing key before any output
  is written.
- Without the `geo_infer` mapping, the five default formats are produced
  exactly as before.

#### Export Options

- `include_metadata` (bool): Include metadata in exports (default: `True`)
- `validate_schema` (bool): Validate XML schema (default: `True`)
- `pretty_print` (bool): Pretty-print JSON/XML (default: `True`)
- `compress` (bool): Compress large exports (default: `False`)

#### Graph Export Configuration

- `graph_layout` (str): Graph layout algorithm (default: `"force"`)
  - Options: `"force"`, `"hierarchical"`, `"circular"`
- `include_weights` (bool): Include edge weights in graphs (default: `True`)
- `node_attributes` (List[str]): Node attributes to include (default: all)

---

## Dependencies

### Required Dependencies

- `json` - JSON export
- `xml.etree.ElementTree` - XML serialization (writers)
- `defusedxml` - XML, GraphML, and GEXF validation reads without DTD/entity expansion
- `pickle` - Pickle serialization

### Optional Dependencies

- `networkx` - Graph format export (recovery: basic XML-based export)

---

## Usage Examples

### Basic Usage

```python
from export import process_export

success = process_export(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/7_export_output"),
    verbose=True,
)
```

### Specific Formats

```python
from export import export_model

results = export_model(
    model_data=parsed_data,
    output_dir=Path("output/7_export_output"),
    formats=["json", "graphml", "gexf"],
)
```

---

## Output Specification

### Output Products

- `{model}_export.json` - JSON export
- `{model}_export.xml` - XML export
- `{model}_graph.graphml` - GraphML graph
- `{model}_graph.gexf` - GEXF graph
- `{model}_model.pkl` - Pickle serialization
- `export_summary.json` - Export summary

### Output Directory Structure

```
output/7_export_output/
├── model_name_export.json
├── model_name_export.xml
├── model_name_graph.graphml
├── model_name_graph.gexf
├── model_name_model.pkl
└── export_summary.json
```

---

## Performance Characteristics

### Latest Execution

- **Duration**: 61ms
- **Memory**: 28.7 MB
- **Status**: SUCCESS_WITH_WARNINGS
- **Formats Generated**: 5

---

## Error Handling

### Graceful Degradation

- **Format Unavailable**: Skip unavailable format, log warning, continue with others
- **Schema Validation Failure**: Export without validation, log warning
- **Large Model**: Use compression or split exports, provide warnings
- **Invalid GNN Model**: Return structured error, skip model

### Error Categories

1. **Format Errors**: Format not supported or unavailable (recovery: skip format)
2. **Validation Errors**: Schema validation fails (recovery: export without validation)
3. **Serialization Errors**: Cannot serialize model (return error)
4. **File I/O Errors**: Cannot write export files (return error)

### Error Recovery

- **Format Recovery**: Automatically skip unavailable formats
- **Partial Export**: Export what's possible, report failures
- **Resource Cleanup**: Proper cleanup of export resources on errors

---

## Integration Points

### Pipeline Integration

- **Input**: Receives parsed GNN models from Step 3 (gnn processing)
- **Output**: Writes multi-format exports to `output/7_export_output/` (no dedicated downstream step consumer; included in the uniform output-dir scans of Steps 20, 22, and 23)
- **Dependencies**: Requires GNN parsing results from `3_gnn.py` output

### Module Dependencies

- **gnn/**: Loads the Step 3 manifest + parsed models from `output/3_gnn_output/`
- **visualization/**: Provides graph formats for visualization
- **render/**: Provides model data for code generation
- **website/**: Provides export data for website generation

### External Integration

- **Cytoscape**: GraphML format for network analysis
- **Gephi**: GEXF format for graph visualization
- **NetworkX**: Graph format conversion and analysis

### Data Flow

```
output/3_gnn_output/ (Step 3 manifest + parsed models)
  ↓
7_export.py (Multi-format export)
  ↓
  └→ output/7_export_output/ (Standalone exports; no dedicated downstream step consumer)
```

---

## Testing

### Test Files

- `tests/export/test_export_overall.py`

### Test Coverage

- Measure: `uv run --extra dev python -m pytest tests/export/ --cov=gnn.export --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

### Key Test Scenarios

1. Multi-format export generation
2. Format validation and error handling
3. Graph format conversion
4. Export integrity verification

---

## MCP Integration

### Tools Registered

- `process_export` — Generate multi-format exports for GNN files in a directory
- `export_single_gnn_file` — Export a single GNN file to all supported formats
- `export.list_functions` — List callable functions in the export module
- `export.call_function` — Call any public function by name with keyword arguments

### MCP File Location

- `src/gnn/export/mcp.py` — Tool registrations and MCP wrappers

---

## Troubleshooting

### Common Issues

#### Issue 1: Export fails for specific format

**Symptom**: Export succeeds for some formats but fails for others  
**Cause**: Missing optional dependency (networkx) or format-specific errors  
**Solution**:

- Check that required dependencies are installed: `uv pip install networkx`
- Use `--verbose` flag to see detailed error messages
- Check format-specific requirements in documentation

#### Issue 2: GraphML/GEXF export fails

**Symptom**: Graph formats fail to generate  
**Cause**: Missing networkx dependency or invalid graph structure  
**Solution**:

- Install networkx: `uv pip install networkx`
- Verify GNN model has valid connections section
- Check that graph data is properly structured

#### Issue 3: Large model export timeout

**Symptom**: Export times out or runs out of memory  
**Cause**: Model too large for single export operation  
**Solution**:

- Use `compress=True` option to reduce file size
- Export formats individually instead of all at once
- Process models in smaller batches

### Performance Issues

#### Slow Export Performance

**Symptoms**: Export takes longer than expected  
**Diagnosis**:

```bash
# Enable verbose logging
uv run python src/gnn/7_export.py --target-dir input/ --verbose
```

**Solutions**:

- Export only needed formats (don't export all formats if not needed)
- Use pickle format for fastest serialization
- Disable schema validation for faster XML export

---

## Version History

### Current module status

**Features**:

- Multi-format export (JSON, XML, GraphML, GEXF, Pickle, Plaintext Summary, Plaintext DSL)
- Format validation and error handling
- Graph format conversion via NetworkX
- Export integrity verification
- MCP tool integration

**Known Issues**:

- None currently

### Roadmap

- **Future**: Streaming export for very large models

---

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [GNN Export Guide](../../../docs/gnn/integration/gnn_export.md)

### External Resources

- [GraphML Specification](http://graphml.graphdrawing.org/)
- [GEXF Format](https://gexf.net/)
- [NetworkX Documentation](https://networkx.org/)

---

**Last Updated**: 2026-01-21
**Maintainer**: GNN Pipeline Team
**Status**: ✅ Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: ✅ 100% Thin Orchestrator Pattern


---
## Documentation
- **[README](../../../src/gnn/export/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/export/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/export/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/export/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/export](../../../src/gnn/export)
