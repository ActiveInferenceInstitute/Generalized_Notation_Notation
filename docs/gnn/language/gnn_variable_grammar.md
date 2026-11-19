# GNN Variable Grammar

**Version**: v2.0.0 documentation bundle\
**Status**: Formal specification

---

## Production Rules

The following EBNF-style grammar defines GNN variable declarations:

```ebnf
declaration  = name "[" dim-list "," type-decl ("," default-decl)? "]" comment? ;
name         = identifier ;
identifier   = (letter | "_" | "π" | "'") (letter | digit | "_" | "'" )* ;
dim-list     = dimension ("," dimension)* ;
dimension    = positive-integer | named-ref ;
positive-int = digit+ ;
named-ref    = identifier ;
type-decl    = "type=" type-value ;
type-value   = "float" | "int" | "bool" ;
default-decl = "default=" default-value ;
default-value = "uniform" | "zeros" | "ones" | "eye" | "random" | numeric-literal ;
comment      = "#" any-char* ;
```

## Valid Examples

```gnn
A[3,3, type=float]                        # 3×3 float matrix
s[num_states, type=float, default=uniform] # named ref, uniform default
π[5, type=float]                          # policy variable (pi)
o[3, type=int]                            # observation (integer)
W[4,4, type=float, default=zeros]         # initialized to zeros
B[3,3,3, type=float, default=ones]        # 3D tensor, ones
```

## Parsing Rules

1. **Whitespace**: Leading/trailing whitespace is stripped. Internal whitespace between tokens is ignored.
2. **Case sensitivity**: `A` ≠ `a`. Variable names are case-sensitive.
3. **Unicode**: `π` (U+03C0) and prime `'` (U+0027) are valid name characters.
4. **Comments**: `#` starts an inline comment; rest of line is ignored.
5. **Named references**: Dimension values like `num_states` are opaque identifiers resolved at validation time.

## Implementation

- Parser: [`src/gnn/schema/parser.py :: parse_state_space()`](../../../src/gnn/schema/__init__.py)
- Validator: [`src/gnn/schema/parser.py :: validate_matrix_dimensions()`](../../../src/gnn/schema/__init__.py)
- CLI: `gnn validate <file.md>` runs all variable grammar checks

---

*See [GNN v1.1 Syntax Specification](../gnn_syntax.md) for the canonical reference.*
