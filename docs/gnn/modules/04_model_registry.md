# Step 4: Model Registry

## Architectural Mapping

**Orchestrator**: `src/gnn/4_model_registry.py` (61 lines)
**Implementation Layer**: `src/gnn/model_registry/`

## Module Description

This module provides comprehensive model registry capabilities for GNN models, including versioning, metadata management, model discovery, and lifecycle management.


```
src/gnn/model_registry/
├── __init__.py                    # Module initialization and exports
├── README.md                      # This documentation
├── AGENTS.md                      # Agent scaffolding documentation
├── SPEC.md                        # Module specification
├── mcp.py                         # Model Context Protocol integration
└── registry.py                    # Core registry functionality
```

## Agent Identity & Capabilities

# Model Registry Module - Agent Scaffolding

## Module Overview

**Purpose**: Centralized model registry for GNN models with versioning, metadata management, and lifecycle tracking

**Pipeline Step**: Step 4: Model registry (4_model_registry.py)

**Category**: Core Management

**Status**: ✅ Production Ready

**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-01-21

---

## Core Functionality

### Primary Responsibilities

1. Register GNN models with unique IDs
2. Track model versions and metadata
3. Manage model lifecycle states
4. Store model relationships and dependencies
5. Enable model discovery and retrieval

### Key Capabilities

- Model registration with automatic ID generation
- Version tracking and history
- Metadata extraction from GNN specifications
- Model search and filtering
- Registry persistence (JSON-based)

---

## API Reference

### Public Functions

#### `process_model_registry(target_dir, output_dir, **kwargs) -> Dict[str, Any]`

**Description**: Main registry processing function. Discovers and registers all GNN files.

**Parameters**:

- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for registry
- `**kwargs`: Additional options

**Returns**: Dictionary with `processed_files`, `successful_registrations`, `registry_path`, and `total_models` counts

**Example**:

```python
from model_registry import process_model_registry

results = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output"),
)
print(
    f"Registered {results['successful_registrations']} of {results['processed_files']} files"
)
```

### Public Classes

#### `ModelRegistry`

**Description**: Main registry class for managing GNN models

**Methods**:

- `register_model(model_path: Path) -> bool` - Register new model
- `get_model(model_id: str) -> Optional[ModelEntry]` - Retrieve model entry
- `search_models(query: str) -> List[ModelEntry]` - Search models by name, description, or tags
- `list_models() -> List[ModelEntry]` - List all registered models
- `delete_model(model_id: str) -> bool` - Delete model from registry
- `save() -> None` - Persist registry to disk
- `load() -> None` - Load registry from disk

**Example**:

```python
registry = ModelRegistry(registry_path)
success = registry.register_model(Path("model.md"))
registry.save()
```

---

## Dependencies

### Required Dependencies

- `json` - Registry persistence
- `pathlib` - File operations
- `datetime` - Timestamp generation

### Internal Dependencies

- `re` - Metadata extraction patterns
- `hashlib` - Model content hashing

---

## Configuration

### Configuration Options

#### Registry Path

- `registry_path` (Path): Custom path for registry file (default: `output_dir / "model_registry.json"`)
- `registry_backup` (bool): Create backup before overwriting (default: `True`)
- `registry_format` (str): Registry format (default: `"json"`)

#### Model Registration Options

- `auto_version` (bool): Automatically increment version on re-registration (default: `True`)
- `extract_metadata` (bool): Extract metadata from GNN files (default: `True`)
- `validate_before_register` (bool): Validate GNN file before registration (default: `True`)

#### Registry Management

- `max_registry_size` (int): Maximum registry size in MB (default: `100`)
- `cleanup_old_versions` (bool): Remove old versions when max size reached (default: `False`)
- `registry_lock` (bool): Use file locking for concurrent access (default: `True`)

---

## Usage Examples

### Basic Usage

```python
from model_registry import process_model_registry

results = process_model_registry(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/4_model_registry_output"),
)
print(f"Registered {results['total_models']} models")
```

### Direct Registry Usage

```python
from model_registry.registry import ModelRegistry

registry = ModelRegistry(Path("registry.json"))
registry.register_model(Path("model.md"))
models = registry.list_models()
registry.save()
```

---

## Output Specification

### Output Products

- `model_registry.json` - Main registry database
- `model_registry_summary.json` - Processing summary
- `{model_id}/` - Per-model metadata directories

### Output Directory Structure

```
output/4_model_registry_output/
├── model_registry.json
├── model_registry_summary.json
└── {model_id}/
    └── {model_id}_metadata.json
```

---

## Performance Characteristics

### Latest Execution

- **Duration**: 56ms
- **Memory**: 28.7 MB
- **Status**: SUCCESS
- **Models Registered**: 1

---

## Registry Schema

### Model Entry Structure

```json
{
  "model_id": "actinf_pomdp_agent",
  "model_name": "Active Inference POMDP Agent",
  "file_path": "input/gnn_files/discrete/actinf_pomdp_agent.md",
  "file_size_bytes": 1759,
  "registered_at": "2025-09-29T12:00:00",
  "version": "1.0.0",
  "metadata": {
    "variables": 13,
    "connections": 11,
    "formats_generated": 23
  }
}
```

---

## Error Handling

### Graceful Degradation

- **Registry File Locked**: Wait and retry, log warning
- **Invalid GNN File**: Skip registration, log error, continue with other files
- **Registry Corruption**: Attempt recovery from backup, log warning
- **Disk Full**: Return error, cannot save registry

### Error Categories

1. **File I/O Errors**: Cannot read/write registry file (recovery: use in-memory registry)
2. **Validation Errors**: Invalid GNN file structure (recovery: skip file, continue)
3. **Version Conflicts**: Model already registered with different version (recovery: auto-increment)
4. **Registry Corruption**: Invalid JSON structure (recovery: restore from backup)

### Error Recovery

- **Backup Restoration**: Automatically restore from backup if registry corrupted
- **Partial Registration**: Register what's possible, report failures
- **Resource Cleanup**: Proper cleanup of registry locks on errors

---

## Integration Points

### Pipeline Integration

- **Input**: Re-globs and re-reads GNN files directly from the target directory (no Step 3 artifacts consumed)
- **Output**: Writes registry JSON and per-model entries to `output/4_model_registry_output/` (plus its own `registry.json` self-state)
- **Dependencies**: None on prior steps' outputs

### Module Dependencies

- **gnn/**: Reads raw GNN file content for registration
- **type_checker/**: Uses registry for model lookup
- **validation/**: Uses registry for model validation
- **report/**: Uses registry for model summaries

### External Integration

- **JSON Storage**: Registry persisted as JSON file
- **File System**: Model metadata stored in directory structure

### Data Flow

```
input/gnn_files (re-globbed and re-read by 4_model_registry.py)
  ↓
4_model_registry.py (Model registration)
  ↓
  └→ output/4_model_registry_output/ (Registry database; no dedicated downstream consumer)
```

---

## Testing

### Test Files

- `tests/model_registry/test_model_registry_overall.py`

### Test Coverage

- Measure: `uv run --extra dev python -m pytest tests/model_registry/ --cov=gnn.model_registry --cov-report=term-missing` (do not treat fixed percentages in this doc as canonical).

---


---
## Documentation
- **[README](../../../src/gnn/model_registry/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/model_registry/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/model_registry/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/model_registry/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/model_registry](../../../src/gnn/model_registry)
