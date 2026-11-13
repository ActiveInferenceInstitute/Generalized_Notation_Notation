# SymPy Documentation Agent

> **📋 Document Metadata**  
> **Type**: Symbolic Mathematics Integration Agent | **Audience**: Researchers, Mathematicians | **Complexity**: Intermediate-Advanced  
> **Cross-References**: [README.md](README.md) | [SymPy GNN Guide](gnn_sympy.md) | [MCP Integration](../mcp/README.md) | [Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md) | [Main Documentation](../README.md)

## Overview

This directory contains comprehensive documentation, resources, and implementation guides for integrating **SymPy** (Symbolic Mathematics Library) with GNN (Generalized Notation Notation). SymPy provides symbolic computation capabilities through the Model Context Protocol (MCP), enhancing mathematical processing for Active Inference model specification, validation, and analysis.

> **Scope note**: SymPy is integrated as an MCP tool surface (`src/gnn/mcp/sympy_mcp.py`, `sympy_mcp_client.py`), not as a render or execution framework. It is not an entry in `src/gnn/render/framework_registry.py`, has no Step 12 executor, and does no continuous linear-Gaussian rendering.

**Status**: MCP tool integration implemented in `src/gnn/mcp/` (8 SymPy tools registered via Step 21); SymPy itself is not a render/execution framework  
**Version**: 1.0

## Purpose

SymPy integration enables:

- **Mathematical Validation**: Validate mathematical expressions in GNN files
- **Expression Simplification**: Simplify and canonicalize GNN equations
- **LaTeX Generation**: Consistent mathematical formatting
- **Symbolic Analysis**: Symbolic manipulation of Active Inference equations
- **MCP Integration**: Model Context Protocol server for AI agent interaction

## Contents

Browse the [current directory](./) for files and subdirectories; use the task links below for navigation.

## Quick Navigation

- **README.md**: [Directory overview](README.md)
- **GNN Documentation**: [gnn/AGENTS.md](../gnn/AGENTS.md)
- **Main Documentation**: [docs/README.md](../README.md)
- **Pipeline Reference**: [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md)

## Integration with Pipeline

This documentation references the 25-step GNN processing pipeline as follows:

The implemented integration point is **Step 21 (MCP)**: SymPy MCP tool registration.
Other touchpoints listed below are **proposals**; no code realizes them yet:

### Implemented

- **Step 21 (MCP)**: SymPy MCP tool registration (`src/gnn/mcp/sympy_mcp.py`)

### Proposed (no runtime)

- **Step 3 (GNN)**: SymPy validation of equation sections
- **Step 5 (Type Checker)**: mathematical expression validation via SymPy
- **Step 11 (Render)** / **Step 12 (Execute)**: no SymPy render target or executor exists


See [src/gnn/AGENTS.md](../../src/gnn/AGENTS.md) for complete pipeline documentation.

## Function Signatures and API (proposed signatures only)

The signatures below are illustrative sketches of a pipeline-level validation API;
the implemented surface is the eight `sympy_*` MCP tools in
`src/gnn/mcp/sympy_mcp.py` (e.g. `sympy_validate_equation`,
`sympy_validate_matrix`, `sympy_simplify_expression`, `sympy_get_latex`).

### Symbolic Computation Functions

```python
def validate_gnn_equations(gnn_equations: List[str]) -> Dict[str, ValidationResult]:
    """
    Validate mathematical expressions from GNN Equations section.

    Parameters:
        gnn_equations: List of equation strings from GNN file

    Returns:
        Dictionary mapping equations to validation results
    """


def simplify_expression(expression: str) -> str:
    """
    Simplify mathematical expression using SymPy.

    Parameters:
        expression: Mathematical expression string

    Returns:
        Simplified expression string
    """
```

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

## See Also

- **[SymPy Cross-Reference](../CROSS_REFERENCE_INDEX.md#sympy)**: Cross-reference index entry
- **[MCP Integration](../mcp/README.md)**: Model Context Protocol
- **[Mathematical Foundations](../gnn/advanced/advanced_modeling_patterns.md)**: Mathematical modeling
- **[Main Index](../README.md)**: Return to main documentation

---

**Status**: MCP tool integration implemented in `src/gnn/mcp/` (8 SymPy tools registered via Step 21); SymPy itself is not a render/execution framework  
**Maintenance**: Regular updates with new SymPy features and integration capabilities
