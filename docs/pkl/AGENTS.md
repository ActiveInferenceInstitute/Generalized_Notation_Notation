# Pkl Documentation Agent

> **📋 Document Metadata**  
> **Type**: Configuration Language Integration Agent | **Audience**: Developers, Configuration Engineers | **Complexity**: Intermediate  
> **Cross-References**: [README.md](README.md) | [Pkl GNN Guide](pkl_gnn.md) | [Configuration Management](../configuration/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **Pkl** (Apple's Configuration Language) with GNN (Generalized Notation Notation). Pkl provides configuration-as-code capabilities with type safety, validation, and multi-format output generation, enhancing GNN model specification and management.

> **Scope note**: Apple Pkl is a *registered serialization format* (`GNNFormat.PKL`, `.pkl` — `PKLParser` in `src/gnn/parsers/schema_parser.py`, `PKLSerializer` in `src/gnn/parsers/pkl_serializer.py`), exercised by Step 3 (`3_gnn.py`) multi-format serialization, with artifacts written to `output/3_gnn_output/`. It is not a render or execution framework: no entry in `src/gnn/render/framework_registry.py`, no Step 11 render target, and no Step 12 executor. Distinct from this, the binary Python-pickle export in `src/gnn/export/` (Step 7) also uses a `.pkl` extension; per the format convention, `.pkl` defaults to textual Pkl DSL and binary pickle files should use `.pickle`.

**Status**: Format support implemented in `src/gnn/parsers/` (PKL parser + serializer); Pkl is not a render/execution framework  
**Version**: 1.0

## Purpose

Pkl integration enables:

- **Enhanced Model Specification**: Type-safe GNN model definitions
- **Multi-Format Export**: Native multi-format rendering capabilities
- **Validation**: Built-in type system with constraint validation
- **Template System**: Reusable Active Inference model templates
- **Configuration-as-Code**: Scientific model configuration management

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

## Quick Navigation

- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [docs/README.md](../README.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)

## Integration with Pipeline

This documentation references the 25-step GNN processing pipeline as follows:

The implemented integration point is the **format registry**, exercised at
**Step 3 (`3_gnn.py`) multi-format serialization** (artifacts in `output/3_gnn_output/`):
`PKLParser` and `PKLSerializer` handle `GNNFormat.PKL`. There is no Pkl involvement
in Steps 5, 11, 12, 17, or 23 — the touchpoints below are **proposals**, not
implemented behavior.

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Function Signatures and API (proposed signatures only)

The signatures below are illustrative sketches; no such functions exist in the
codebase. The implemented Pkl surface is `PKLParser`/`PKLSerializer` in
`src/gnn/parsers/`.

### Configuration Management Functions

```python
def generate_pkl_config(gnn_model: GNNModel) -> PklConfig:
    """
    Generate Pkl configuration from GNN model.

    Parameters:
        gnn_model: Parsed GNN model structure

    Returns:
        PklConfig with type-safe model definition
    """


def validate_pkl_config(config: PklConfig) -> ValidationResult:
    """
    Validate Pkl configuration using built-in type system.

    Parameters:
        config: Pkl configuration to validate

    Returns:
        ValidationResult with type checking results
    """
```

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Configuration Management](../configuration/README.md)**: Configuration systems

### Configuration Resources
- **[Export Formats](../export/README.md)**: Multi-format export
- **[Type Checking](../../src/gnn/type_checker/AGENTS.md)**: Type validation
- **[Configuration Management](../configuration/README.md)**: Configuration systems

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## See Also

- **[Pkl Cross-Reference](../CROSS_REFERENCE_INDEX.md#pkl)**: Cross-reference index entry
- **[Configuration Management](../configuration/README.md)**: Configuration systems
- **[Export Formats](../export/README.md)**: Multi-format export
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: Format support implemented in `src/gnn/parsers/` (PKL parser + serializer at Step 3 serialization); Pkl is not a render/execution framework  
**Maintenance**: Regular updates with new Pkl features and integration capabilities
