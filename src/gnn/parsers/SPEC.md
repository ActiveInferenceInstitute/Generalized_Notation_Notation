# Parsers — specification

## Role

`src/gnn/parsers/` implements multi-format **parse** and **serialize** for GNN models: one pair of modules per family of formats, plus shared infrastructure.

## Registries

- **`system.py`** — `PARSER_REGISTRY` (23 entries), `SERIALIZER_REGISTRY` (22 entries), class **`GNNParsingSystem`**. PNML has a parser but no serializer registry entry; see **[../SPEC.md](../SPEC.md)**.
- **`common.py`** — **`GNNFormat`** enum, **`GNNParser`** protocol, internal representation types, shared helpers.

## Layout

| Pattern | Purpose |
|---------|---------|
| `*_parser.py` | Format-specific parsing |
| `*_serializer.py` | Format-specific output |
| `grammar_parser.py` / `grammar_serializer.py` | BNF / EBNF |
| `schema_parser.py` / `schema_serializer.py` | XSD, ASN.1, PKL, Alloy, Z (parsers vary) |
| `xml_parser.py` / `xml_serializer.py` | XML and PNML parsing |
| `unified_parser.py` | Optional unified entry |
| `validators.py` | Parser-oriented validation helpers |

## Requirements

- **Python** >= 3.11 (project `requires-python`; see repo root `pyproject.toml`).
- Optional extras (e.g. protobuf) may be required for some formats at runtime.
- Authored PKL class properties are parsed through complete line boundaries:
  the full declared type and optional default value remain distinct, including
  scalar and generic types. Type capture must not stop at its first character.
- XML discovery may find one physical element through multiple container and
  generic XPath paths. Each element contributes once, in discovery priority;
  separate authored declarations remain separate even when their names match.
- ASN.1 and Z native `MODEL_DATA` line comments admit a JSON object from that
  line only. Later declarations cannot extend its payload or replace its model
  identity; present malformed/non-object data fails with a causal diagnostic.
  Absent metadata retains native declaration parsing, and existing ASN.1 block
  comment payloads remain supported.
- Alloy and Z saved-payload reconstruction preserves a supplied parameter
  description without adding an interpretation or changing its value.

## Testing

Primary coverage lives under `tests/test_gnn*.py` and `src/gnn/testing/`. Run e.g. `uv run --extra dev python -m pytest tests/gnn/test_gnn_parsing.py -q`.
