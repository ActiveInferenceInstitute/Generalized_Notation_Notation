# Ontology Module Specification

## Overview
Ontology processing and validation for GNN models.

## Components

### Core
- `processor.py` - Ontology processor (pure composable primitives + pipeline entry points)
- `utils.py` - Module metadata helpers (`get_module_info`, `get_ontology_processing_options`, `get_mcp_interface`)
- `mcp.py` - MCP tool registration (4 tools, validated against the real vocabulary)
- `act_inf_ontology_terms.json` - Active Inference ontology vocabulary (64 declared labels; `uri` values are declared metadata, never emitted as identifiers)
- `bridge.py` - Pinned vocabulary bridge: copy/pin verification, declared-label index, identifier resolution
- `active_inference_ontology_terms.json` + `ontology-pin.json` - Pinned Active Inference Ontology vocabulary bridge copy (schema `active-inference-ontology/gnn-terms/v1`, release v5) and its repository/commit/SHA-256 pin

## Features
- Ontology term validation (case-insensitive)
- Semantic mapping
- Term extraction
- Nearest-term suggestion (Levenshtein + substring)
- In-memory vocabulary construction
- Prebuilt batch vocabulary index (`OntologyTermIndex`)
- Injectable vocabulary search paths (dependency injection)
- Pinned vocabulary bridge resolution (release-v5 `term-*` identifiers; labels without a counterpart report `not in ontology release v5`)

## Key Exports
```python
from gnn.ontology import (
    process_ontology,
    analyze_ontology_content,
    validate_annotations,
    suggest_terms,
    summarise_coverage,
    build_ontology_terms,
    parse_annotation,
    ParsedAnnotation,
)
```


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
