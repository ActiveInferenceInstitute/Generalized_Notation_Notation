# Specification: GNN Reference Documentation

## Scope
Working reference material for the GNN language: file structure, type
system, schema, and style rules, with runnable examples and validation
commands. The **normative** specification is
[`docs/gnn/gnn_syntax.md`](../gnn_syntax.md); where this subtree and the
specification disagree, the specification wins and the file here is the bug.

## Contents
| File | Purpose |
|------|---------|
| `gnn_syntax.md` | Practical syntax companion — section obligation table, worked examples, parameterization families |
| `gnn_file_structure_doc.md` | GNN file anatomy: section order, metadata, expected headings |
| `gnn_type_system.md` | Variable type annotations (int, float, bool, matrix, tensor) |
| `gnn_schema.md` | JSON Schema for parsed GNN dicts |
| `gnn_standards.md` | Style and naming conventions |
| `gnn_dsl_manual.md` | DSL symbols and operators table |
| `technical_reference.md` | Deep-dive technical reference |
| `architecture_reference.md` | Pipeline architecture overview |

## Versioning
- GNN language version: tracked in [`docs/gnn/gnn_syntax.md`](../gnn_syntax.md) (v1.1)
- Document bundle version: inherits from [`docs/gnn/SPEC.md`](../SPEC.md)
- Package version: 4.1.0 (matches `pyproject.toml`)

## Status
Maintained. Every file in this subtree is referenced from `docs/gnn/README.md`
and should remain consistent with `src/gnn/schema/parser.py` and
`src/gnn/parsers/markdown_parser.py`. A change to the normative syntax must
update [`docs/gnn/gnn_syntax.md`](../gnn_syntax.md), the parser, and the
affected pages here in the same change.
