# GNN parsers

Multi-format **parse** and **serialize** for GNN models: one `*_parser.py` / `*_serializer.py` pair per format family, plus shared infrastructure.

## Entry points

- **`GNNParsingSystem`** in `system.py` — the registry-driven system: `parse_file`, `serialize`, format conversion.
- **`gnn.parsers.basic`** — the structural/formal parse surface (`GNNFormalParser`, `parse_gnn_formal`, `validate_gnn_syntax`, `GNNFormatSpec`); its own `GNNParsingSystem` is reachable only as `gnn.parsers.basic.GNNParsingSystem` to avoid colliding with the registry-driven one.
- **`frontmatter.py`** — `parse_frontmatter` / `has_frontmatter` for YAML frontmatter blocks.
- **`PARSER_REGISTRY`** / **`SERIALIZER_REGISTRY`** — map `GNNFormat` to concrete classes.
- **`GNNFormat`**, **`GNNParser`** (protocol) — `common.py`.

Canonical **enum size**, registry counts, and round-trip scope: **[../SPEC.md](../SPEC.md)**.

## Layout

See **[SPEC.md](SPEC.md)** for the full layout table (`grammar_*`, `schema_*`, `xml_*`, `unified_parser.py`, `validators.py`, etc.).

## Connection annotations

`Connection.annotation` preserves the label after a connection's colon without changing source or target names. JSON/YAML interchange and embedded model dictionaries serialize this optional field, and reconstruction restores it. Older dictionaries that omit it retain `None`. The focused regression `tests/gnn/test_connection_annotation_roundtrip.py` exercises absent and Unicode labels across every registered serializer/parser pair; this is label-fidelity evidence, not a claim of complete model fidelity for every format.

## Pickle inputs

Pickle inputs use `safe_pickle_load` and `safe_pickle_loads` in
`binary_parser.py`. Both inspect the same bytes they reconstruct, reject
extension opcodes that could bypass the global allowlist, and reject bytes
after the first record. Parsing, schema validation and export validation use
this shared boundary. It limits object reconstruction; it does not authenticate
artifacts or bound memory and CPU use.

## Saved schema interchange

ASN.1 `-- MODEL_DATA:` and Z `% MODEL_DATA:` comments carry a JSON object on
one native comment line. Reading that payload stops at the line boundary, so
later native braces cannot replace the saved model identity. When the line
comment supplies the interchange payload, malformed or non-object data produces
a failed parse with a causal diagnostic; absent metadata continues through the
native declaration reader. ASN.1's existing block comment interchange and its
extraction priority remain supported.

Saved schema payloads retain supplied parameter values. Alloy and Z restore
parameter descriptions when the payload includes them; older payloads may omit
that field. These interchange guarantees do not establish formal equivalence
between arbitrary native schemas and the source GNN model.

## Markdown datatype fields

Bracket declarations accept comma-separated fields with surrounding whitespace,
including `reported[2, 1, type=int]`. The Markdown registry binds the trailing
`type=` field independently of adjacent annotation fields. `int` becomes the
existing `integer` datatype and documented `bool` becomes `binary`; the existing
`binary` spelling remains supported. Absent types retain the float default and
unrecognized types retain the categorical recovery. Original annotation text,
including default hints, remains in `raw_sections`; parsing does not initialize
new values from a hint.

Nested JSON list values containing lowercase booleans retain their exact shape
and scalar values. This array recovery reuses the literal parser's existing
10,000-character and depth-10 guards before decoding the untouched JSON source.
It does not add JSON mapping admission or change tuple, malformed-value or
overlimit legacy recovery.

## Adding a format

1. Extend **`GNNFormat`** in `common.py` if needed.
2. Implement parser and (unless parse-only) serializer classes.
3. Register in **`PARSER_REGISTRY`** and **`SERIALIZER_REGISTRY`** in **`system.py`**.
4. Add tests under `tests/` and extend `tests/testing/test_round_trip.py` (participation configured in `src/gnn/testing/round_trip_config.py`) if the format should join the default round-trip list.

## Tests

```bash
uv run --extra dev python -m pytest tests/gnn/test_gnn_parsing.py tests/gnn/test_gnn_parsers_common.py -q
```

Agent-oriented detail: **[AGENTS.md](AGENTS.md)**.
