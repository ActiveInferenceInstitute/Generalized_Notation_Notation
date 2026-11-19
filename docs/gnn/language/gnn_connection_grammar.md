# GNN Connection Grammar

**Version**: v2.0.0 documentation bundle\
**Status**: Formal specification

---

## Production Rules

```ebnf
connection   = source edge-op target (":" annotation)? comment? ;
source       = identifier ;
target       = identifier ;
edge-op      = ">" | "-" ;
annotation   = (letter | digit | "_")+ ;
identifier   = (letter | "_" | "π" | "'") (letter | digit | "_" | "'" )* ;
comment      = "#" any-char* ;
```

## Edge Types

| Operator | Meaning | Example | Semantics |
|----------|---------|---------|-----------|
| `>` | Directed | `A>B` | A causally influences B |
| `-` | Undirected | `A-B` | Bidirectional association |

## Annotations (v1.1)

Annotations are optional labels after a colon:

```gnn
D>s:prior_initialization     # prior belief over hidden states
A-o:observation_mapping       # likelihood mapping
G>π:policy_selection          # expected free energy → policy
s>o:emission                  # state to observation emission
```

### Annotation Rules

1. Annotations must match `[a-zA-Z0-9_]+` (alphanumeric + underscore).
2. Parsers **must** preserve annotations but **may** ignore them for structural validation.
3. Annotations serve as labels for rendering, documentation, and editor hover.

## Validation Rules

| Rule | Code | Severity | Description |
|------|------|----------|-------------|
| Known source | `GNN-W002` | warning | Source variable is not declared in `StateSpaceBlock` |
| Known target | `GNN-W002` | warning | Target variable is not declared in `StateSpaceBlock` |
| Parseable syntax | `GNN-E005` | error | Line does not match the connection grammar |

An undeclared endpoint is reported as a **warning**, not an error: parsing
continues and the edge is kept. `GNN-E003` is reserved for this condition in
the error taxonomy but has no emitting code — `parse_connections` in
[`src/gnn/schema/parser.py`](../../../src/gnn/schema/__init__.py) raises `GNN-W002` for both
endpoints instead. Cross-validation only happens when the parser is given the
declared-variable set; parsing a `Connections` block in isolation reports
neither code.

## Implementation

- Parser: [`src/gnn/schema/parser.py :: parse_connections()`](../../../src/gnn/schema/__init__.py)
- LSP diagnostics: [`src/gnn/lsp/__init__.py`](../../../src/gnn/lsp/__init__.py) — real-time connection error highlighting
- CLI: `gnn validate <file.md>` runs all connection grammar checks

---

*See [GNN v1.1 Syntax Specification](../gnn_syntax.md) for the canonical reference.*
