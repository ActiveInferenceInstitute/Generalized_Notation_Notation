# Parsers Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/parsers/`.

## Purpose
- Validate the public parsing surface: frontmatter helpers, markdown parsing,
  the parse-tree outline, and the format registries.
- Keep tests aligned with `src/gnn/parsers/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/parsers/ -q`.

`test_parsers_public_api.py` exercises `has_frontmatter`, `parse_frontmatter`,
`MarkdownGNNParser.parse_string` (including empty-content error reporting) and
`get_parse_tree_visualization` on small deterministic GNN inputs.

`test_parsers_registry.py` checks `PARSER_REGISTRY` / `SERIALIZER_REGISTRY`
entries and `GNNParsingSystem` format listing plus string parsing.

`test_markdown_section_headings.py` covers section-heading spelling: the spaced
`## ActInf Ontology Annotation` spelling reaches the ontology parser with a
visible warning, an unrecognized `ActInf…` heading warns instead of leaving
silently empty bindings, canonical headings stay warning-free, and whitespace
normalization applies to the other canonical sections too.
