# Schemas — specification

## Role

Schema **source files** for validating or describing serialized GNN models:

| File | Format |
|------|--------|
| `json.json` | JSON Schema |
| `yaml.yaml` | YAML-oriented schema / guidance |
| `proto.proto` | Protocol Buffers |
| `xsd.xsd` | XML Schema |
| `asn1.asn1` | ASN.1 |
| `pkl.pkl` | Pkl |

Runtime parsing and validation use **`src/gnn/parsers/schema_parser.py`**, **`schema_serializer.py`**, and related code in **`src/gnn/parsers/`**.

## Requirements

- **Python** >= 3.12 (see repo `pyproject.toml`).
