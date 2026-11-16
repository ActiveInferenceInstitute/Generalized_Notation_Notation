# Types Module Specification

## Overview
Single authoritative home for the shared dataclass types used across the GNN pipeline — the parsed-document model plus validation and round-trip results — kept dependency-light so any consumer can import it safely.

## Components
- `definitions.py` defines `ValidationLevel`, `GNNSyntaxError`, `ValidationResult`, `GNNVariable`, `GNNConnection`, `ParsedGNN`, `RoundTripResult`, and `ComprehensiveTestReport`. It exposes the parser-owned `ParseResult` as a lazy re-export.
- `__init__.py` exports eleven public names. `GNNFormat`, `GNNInternalRepresentation`, and `ParseResult` resolve lazily from `gnn.parsers.common`.

## Invariants
- Importing the shared dataclasses uses only the standard library. `gnn.parsers.basic` imports its shared types directly from `definitions.py`.
- Each parser alias resolves to the same class object as its original declaration, preserving fully qualified names used by serialized data.

## Key Exports
```python
from gnn.types import (
    ComprehensiveTestReport,
    GNNConnection,
    GNNFormat,
    GNNInternalRepresentation,
    GNNSyntaxError,
    GNNVariable,
    ParseResult,
    ParsedGNN,
    RoundTripResult,
    ValidationLevel,
    ValidationResult,
)
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/types/test_definitions.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
