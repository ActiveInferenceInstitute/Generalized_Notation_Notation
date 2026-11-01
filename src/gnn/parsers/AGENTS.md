# Parsers — agent notes

## Purpose

Multi-format I/O for GNN: **23** `GNNFormat` values, **23** parsers, **22** serializers in **`system.py`** (see **[../SPEC.md](../SPEC.md)**).

## Entry points

- **`GNNParsingSystem`** — the registry-driven system in **`system.py`**; instantiate and call `parse_file`, `serialize`, etc.
- **`gnn.parsers.basic`** — structural/formal parse surface (`GNNFormalParser`, `parse_gnn_formal`, `validate_gnn_syntax`, `GNNFormatSpec`). Its same-named `GNNParsingSystem` is not re-exported from the package; use `gnn.parsers.basic.GNNParsingSystem` explicitly.
- **`frontmatter.py`** — `parse_frontmatter` / `has_frontmatter`.
- **`PARSER_REGISTRY` / `SERIALIZER_REGISTRY`** — maps `GNNFormat` to concrete classes.
- **`GNNFormat`**, **`GNNParser`** (protocol) — **`common.py`**.

## Section headings

`MarkdownGNNParser` matches `## ` headings against its canonical section names.
A heading that differs only by whitespace — `## ActInf Ontology Annotation` for
`## ActInfOntologyAnnotation`, `## Model Name` for `## ModelName` — is
normalized to the canonical name, parsed under it, and reported through
`ParseResult.warnings` plus a `logger.warning`. A heading that names no section
but looks ontological (`ActInf…`, or any `…ontology…` spelling) also warns that
it produced no ontology bindings, so unrecognized spellings never leave
bindings silently empty. Raw heading text stays available in
`model.raw_sections`. The formal `schema_validator.GNNParser` remains
spelling-exact and reports the missing canonical section instead.

## Adding a format

1. Extend **`GNNFormat`** in `common.py` if needed.
2. Implement parser (and serializer unless parse-only) classes.
3. Register in **`PARSER_REGISTRY`** and, if applicable, **`SERIALIZER_REGISTRY`** in **`system.py`**.
4. Add tests under `tests/` and extend `tests/testing/test_round_trip.py` (participation configured in `src/gnn/testing/round_trip_config.py`) if the format should participate in round-trip.

## Tests

```bash
uv run --extra dev python -m pytest tests/gnn/test_gnn_parsing.py tests/gnn/test_gnn_parsers_common.py -q
```
