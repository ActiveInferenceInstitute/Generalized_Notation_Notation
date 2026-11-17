"""Pin the single canonical GNN section contract (W2-01).

The formal schema artifacts (``schemas/json.json`` / ``schemas/yaml.yaml``),
the compliance gates (``schema_validator``, ``validation.mcp``), and the
type_checker display surfaces must all derive from
``gnn.schemas.section_contract`` — editing a schema file or hardcoding a
required-section list in code without updating the table must fail here.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from gnn.schemas.section_contract import (
    CANONICAL_GNN_SECTIONS,
    JSON_SECTION_KEYS,
    OPTIONAL_SECTIONS,
    REQUIRED_SECTIONS,
)

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "src" / "gnn" / "schemas"


def test_required_sections_match_json_schema() -> None:
    schema = json.loads((SCHEMAS_DIR / "json.json").read_text(encoding="utf-8"))
    expected = sorted(JSON_SECTION_KEYS[s] for s in REQUIRED_SECTIONS)
    assert sorted(schema["required"]) == expected


def test_required_sections_match_yaml_schema() -> None:
    schema = yaml.safe_load((SCHEMAS_DIR / "yaml.yaml").read_text(encoding="utf-8"))
    assert list(schema["required_sections"]) == list(REQUIRED_SECTIONS)
    assert list(schema["optional_sections"]) == list(OPTIONAL_SECTIONS)


def test_classification_partitions_canonical_universe() -> None:
    assert set(REQUIRED_SECTIONS) <= set(CANONICAL_GNN_SECTIONS)
    assert set(REQUIRED_SECTIONS).isdisjoint(OPTIONAL_SECTIONS)
    # Every canonical section is classified exactly once (ImageFromPaper is
    # a recognised optional section outside the 13-section canonical order).
    assert set(CANONICAL_GNN_SECTIONS) <= (
        set(REQUIRED_SECTIONS) | set(OPTIONAL_SECTIONS)
    )
    assert JSON_SECTION_KEYS.keys() >= (
        set(CANONICAL_GNN_SECTIONS) | set(OPTIONAL_SECTIONS)
    )


def test_compliance_surface_uses_contract_for_signature_time_fixture() -> None:
    """A Signature-present/Time-absent model must fail compliance with
    ``Time`` (required) — never ``Signature`` (optional) — and the same
    verdict must come from the schema-validator markdown gate."""
    from gnn.validation.mcp import check_schema_compliance_mcp

    content = (
        "## GNNSection\nActInfPOMDP\n\n"
        "## GNNVersionAndFlags\nGNN v1\n\n"
        "## ModelName\nSigNoTime\n\n"
        "## ModelAnnotation\nfixture\n\n"
        "## StateSpaceBlock\ns[2,1,type=float]\n\n"
        "## Connections\ns-s\n\n"
        "## InitialParameterization\ns={}\n\n"
        "## Footer\nSigNoTime\n\n"
        "## Signature\nabc123\n"
    )
    result = check_schema_compliance_mcp(content)
    assert result["success"] is True
    assert result["is_compliant"] is False
    assert result["missing_required"] == ["Time"]
    assert result["unrecognised_sections"] == []

    import tempfile

    from gnn.schema_validator import GNNValidator

    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as handle:
        handle.write(content)
        fixture_path = Path(handle.name)
    try:
        validation = GNNValidator().validate_file(fixture_path)
    finally:
        fixture_path.unlink(missing_ok=True)
    assert any("Time" in str(error) for error in validation.errors)


def test_schema_validator_uses_shared_required_table() -> None:
    import inspect

    from gnn.schema_validator import structural_checks
    from gnn.schema_validator import validator as sv

    # GNNValidator's required-section check lives in StructuralChecksMixin,
    # which must read the shared table; neither module may hardcode it.
    assert issubclass(sv.GNNValidator, structural_checks.StructuralChecksMixin)
    checks_source = inspect.getsource(structural_checks)
    assert "REQUIRED_SECTIONS" in checks_source
    for source in (checks_source, inspect.getsource(sv)):
        assert '"InitialParameterization"' not in source
