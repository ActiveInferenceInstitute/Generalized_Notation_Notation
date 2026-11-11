# Pkl Integration for GNN

> **📋 Document Metadata**  
> **Type**: Configuration Language Integration Guide | **Audience**: Developers, Configuration Engineers | **Complexity**: Intermediate  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [Pkl GNN Guide](pkl_gnn.md) | [Configuration Management](../configuration/README.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating **Pkl** (Apple's Configuration Language) with GNN (Generalized Notation Notation). Pkl provides configuration-as-code capabilities with type safety, validation, and multi-format output generation, enhancing GNN model specification and management.

> **Scope note**: Apple Pkl is a *registered serialization format*, not a render or execution framework. The tree supports `GNNFormat.PKL` (`.pkl`) with `PKLParser` (`src/gnn/parsers/schema_parser.py`) and `PKLSerializer` (`src/gnn/parsers/pkl_serializer.py`), exercised by Step 3 (`3_gnn.py`) multi-format serialization (artifacts in `output/3_gnn_output/`). Pkl is **not** an entry in `src/gnn/render/framework_registry.py`, has no Step 12 executor, and does no continuous linear-Gaussian rendering. Note: `.pkl` is treated as textual Pkl DSL by default; binary Python pickle files use the `.pickle` extension (`GNNFormat.PICKLE`), and the Step 7 pickle export in `src/gnn/export/` is a separate capability that coincidentally also uses a `.pkl` extension.

**Status**: Format support implemented in `src/gnn/parsers/` (PKL parser + serializer); Pkl is not a render/execution framework  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[pkl_gnn.md](pkl_gnn.md)**: Complete Pkl-GNN integration guide
- **[pkl_info.md](pkl_info.md)**: Pkl framework information

### Main Documentation
- **[docs/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[Configuration Management](../configuration/README.md)**: Configuration systems
- **[Export Formats](../../src/gnn/export/README.md)**: Multi-format export
- **[Type Checking](../../src/gnn/type_checker/AGENTS.md)**: Type validation

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)**: Implementation details

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

### Core Files

- **`pkl_gnn.md`**: Complete Pkl-GNN integration guide
  - Configuration-as-code for scientific models
  - Type-safe model definitions
  - Multi-format output generation
  - Built-in validation and type safety

- **`pkl_info.md`**: Pkl framework information
  - Pkl language overview
  - Configuration capabilities

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

### Subdirectories

- **`examples/`**: Pkl configuration examples
- Additional Pkl resources

## Pkl Overview

Pkl provides:

### Configuration-as-Code
- **Type-Safe Definitions**: Compile-time validation of model specifications
- **Template Inheritance**: Shared Active Inference patterns
- **Immutable Configurations**: Preventing accidental model corruption
- **Late Binding Properties**: Complex interdependencies

### Key Features
- **Multi-Format Output**: Native rendering to JSON, XML, GraphML, and more
- **Built-in Validation**: Robust type system with constraints
- **Template System**: Reusable configuration templates
- **Scientific Modeling**: Configuration-as-code for scientific models

## Integration with GNN

Pkl integration enables:

- **Enhanced Model Specification**: Type-safe GNN model definitions
- **Multi-Format Export**: Native multi-format rendering capabilities
- **Validation**: Built-in type system with constraint validation
- **Template System**: Reusable Active Inference model templates

## Integration with Pipeline

Pkl support in the codebase today is **parsing and serialization only**:

1. **Parsing**: `PKLParser` reads textual Pkl DSL input (`.pkl`, `GNNFormat.PKL`).
2. **Serialization**: `PKLSerializer` writes GNN models to Pkl configuration syntax;
   it is one of the 22 registered serializers, called from Step 3 (`3_gnn.py`),
   producing artifacts in `output/3_gnn_output/`.
3. **No render/execution path**: Pkl is not in `framework_registry.py` (no Step 11
   render target) and has no Step 12 executor. The configuration-as-code workflows
   described in [pkl_gnn.md](pkl_gnn.md) are proposals; no code realizes them yet.


See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

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

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with configuration management foundations
- **Functionality**: Describes actual Pkl integration capabilities
- **Completeness**: Comprehensive coverage of configuration-as-code integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[Pkl Cross-Reference](../CROSS_REFERENCE_INDEX.md#pkl)**: Cross-reference index entry
- **[Configuration Management](../configuration/README.md)**: Configuration systems
- **[Export Formats](../export/README.md)**: Multi-format export
- **[Main Index](../README.md)**: Return to main documentation

## Documents in this folder

- [Pkl Examples](examples/README.md)

---

**Status**: Format support implemented in `src/gnn/parsers/` (PKL parser + serializer at Step 3 serialization); Pkl is not a render/execution framework  
**Maintenance**: Regular updates with new Pkl features and integration capabilities
