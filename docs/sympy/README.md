# SymPy Integration for GNN

> **📋 Document Metadata**  
> **Type**: Symbolic Mathematics Integration Guide | **Audience**: Researchers, Mathematicians | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [AGENTS.md](AGENTS.md) | [SymPy GNN Guide](gnn_sympy.md) | [MCP Integration](../mcp/README.md) | [Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md) | [Main Documentation](../README.md)

## Overview

This directory contains documentation, resources, and implementation guides for integrating **SymPy** (Symbolic Mathematics Library) with GNN (Generalized Notation Notation). SymPy provides symbolic computation capabilities through the Model Context Protocol (MCP), enhancing mathematical processing for Active Inference model specification, validation, and analysis.

> **Scope note**: SymPy is a *symbolic-mathematics via MCP* capability, not a render or execution framework. It is not an entry in `src/gnn/render/framework_registry.py` (pymdp, rxinfer, activeinference_jl, jax, discopy, pytorch, numpyro, stan, bnlearn), has no Step 12 executor, and does no continuous linear-Gaussian rendering. The implemented integration lives in `src/gnn/mcp/sympy_mcp.py` and `src/gnn/mcp/sympy_mcp_client.py`.

**Status**: MCP tool integration implemented in `src/gnn/mcp/` (8 SymPy tools registered via Step 21); SymPy itself is not a render/execution framework  
**Version**: 1.0

## Quick Navigation

### This Directory
- **[README.md](README.md)**: Directory overview (this file)
- **[AGENTS.md](AGENTS.md)**: Technical documentation and agent scaffolding
- **[gnn_sympy.md](gnn_sympy.md)**: Complete SymPy-GNN integration guide

### Main Documentation
- **[docs/README.md](../README.md)**: Main documentation hub
- **[CROSS_REFERENCE_INDEX.md](../CROSS_REFERENCE_INDEX.md)**: Complete cross-reference index
- **[learning_paths.md](../learning_paths.md)**: Learning pathways

### Related Directories
- **[MCP Integration](../mcp/README.md)**: Model Context Protocol
- **[Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md)**: Mathematical modeling
- **[Type Checking](../../src/gnn/type_checker/AGENTS.md)**: Type validation

### Pipeline Integration
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)**: Implementation details

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

### Core Files

- **`gnn_sympy.md`**: Complete SymPy-GNN integration guide
  - Symbolic mathematics for Active Inference
  - Expression validation and simplification
  - LaTeX generation and formatting
  - MCP server integration

- **`AGENTS.md`**: Technical documentation and agent scaffolding
  - Complete documentation structure
  - Integration with pipeline
  - Cross-references and navigation

- **`README.md`**: Directory overview (this file)

## SymPy Overview

SymPy provides:

### Symbolic Computation
- **Expression Parsing**: Parse and validate mathematical expressions
- **Symbolic Simplification**: Canonicalize mathematical relationships
- **LaTeX Generation**: Consistent mathematical formatting
- **MCP Integration**: Model Context Protocol server for AI agent interaction

### Key Features
- **Mathematical Validation**: Validate GNN equation sections
- **Expression Simplification**: Simplify complex mathematical expressions
- **LaTeX Support**: Generate LaTeX representations
- **Symbolic Algebra**: Symbolic manipulation of mathematical expressions

## Integration with GNN

SymPy integration enables:

- **Mathematical Validation**: Validate mathematical expressions in GNN files
- **Expression Simplification**: Simplify and canonicalize GNN equations
- **LaTeX Generation**: Consistent mathematical formatting
- **Symbolic Analysis**: Symbolic manipulation of Active Inference equations

## Integration with Pipeline

This documentation references the 25-step GNN processing pipeline as follows:

The implemented integration point is **Step 21 (MCP Processing, `21_mcp.py`)**: the
eight `sympy_*` tools are registered through `src/gnn/mcp/sympy_mcp.py` and are
discoverable by any MCP client. SymPy is also listed as an optional dependency of
the `research` extras group. Other pipeline touchpoints mentioned in the guides
below (equation validation inside Steps 3/5/6, symbolic render-time analysis) are
**proposals**; no code realizes them yet.

See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Related Resources

### Main GNN Documentation
- **[GNN Overview](../gnn/gnn_overview.md)**: Core GNN concepts
- **[GNN Quickstart](../gnn/tutorials/quickstart_tutorial.md)**: Getting started guide
- **[Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md)**: Mathematical modeling

### Mathematical Resources
- **[MCP Integration](../mcp/README.md)**: Model Context Protocol
- **[Type Checking](../../src/gnn/type_checker/AGENTS.md)**: Type validation
- **[Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md)**: Mathematical modeling

### Pipeline Architecture
- **[Pipeline Documentation](../gnn/operations/gnn_tools.md)**: Complete pipeline guide
- **[Pipeline AGENTS](../../src/gnn/AGENTS.md)**: Implementation details
- **[Pipeline README](../../src/gnn/README.md)**: Pipeline overview

## Standards and Guidelines

All documentation in this module adheres to professional standards:

- **Clarity**: Concrete, technical writing with symbolic mathematics foundations
- **Functionality**: Describes actual SymPy integration capabilities
- **Completeness**: Comprehensive coverage of symbolic computation integration
- **Consistency**: Uniform structure and style with GNN documentation ecosystem

## See Also

- **[SymPy Cross-Reference](../CROSS_REFERENCE_INDEX.md#sympy)**: Cross-reference index entry
- **[MCP Integration](../mcp/README.md)**: Model Context Protocol
- **[Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md)**: Mathematical modeling
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: MCP tool integration implemented in `src/gnn/mcp/` (8 SymPy tools registered via Step 21); SymPy itself is not a render/execution framework  
**Maintenance**: Regular updates with new SymPy features and integration capabilities


## Documents in this folder

- [SymPy MCP Integration Demonstration](sympy_integration_demo.md)
- [SymPy MCP Integration Implementation Summary](implementation_summary.md)
