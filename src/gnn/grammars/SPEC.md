# Grammars — specification

## Role

Static **BNF** and **EBNF** grammars for GNN surface syntax:

- `bnf.bnf` — BNF grammar
- `ebnf.ebnf` — EBNF grammar

Used as reference for tokenizer/parser behavior and documentation; runtime parsing of GNN files is driven by **`src/gnn/parsers/grammar_parser.py`** and the Markdown pipeline.

## Content expectations

Grammars include Unicode support (e.g. π in policy names) and comment rules aligned with reference models. See **[README.md](README.md)** for details.

## Requirements

- **Python** >= 3.12 for any tooling that loads these alongside the GNN package (see repo `pyproject.toml`).
