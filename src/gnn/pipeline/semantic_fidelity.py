"""Semantic fidelity gates for maintained GNN model families."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from gnn.parsers import GNNFormat, GNNParsingSystem
from gnn.pipeline.model_family_acceptance import (
    ModelFamily,
    load_model_family_manifest,
    select_model_families,
)
from gnn.report.semantic_fidelity import render_semantic_fidelity_markdown
from gnn.schema import (
    parse_connections,
    parse_state_space,
    validate_matrix_dimensions,
)

STRICT_ROUND_TRIP_FORMATS = ("json",)
SCHEMA_VERSION = "gnn_semantic_fidelity_ledger_v1"
CONTRACT_SCHEMA_VERSION = "gnn_semantic_contract_v2"


@dataclass(frozen=True)
class SemanticDifference:
    """One strict semantic contract mismatch."""

    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Return a serializable difference record."""
        return {"field": self.field, "message": self.message}


def run_semantic_fidelity_gate(
    manifest_path: Path,
    output_dir: Path,
    *,
    family_names: Iterable[str] | None = None,
    formats: Sequence[str] = STRICT_ROUND_TRIP_FORMATS,
    strict: bool = False,
) -> dict[str, Any]:
    """Run strict parse -> serialize -> parse fidelity checks."""
    output_dir.mkdir(parents=True, exist_ok=True)
    families = _select_families(manifest_path, family_names)
    ledger: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(),
        "manifest": str(manifest_path),
        "strict": strict,
        "formats": list(formats),
        "family_count": len(families),
        "families": [],
    }

    failures: list[str] = []
    for family in families:
        family_result = _run_family_fidelity(family, formats, output_dir)
        ledger["families"].append(family_result)
        if family_result["status"] == "failed":
            failures.append(family.name)

    ledger["status"] = "failed" if failures else "passed"
    ledger["failed_families"] = failures
    _write_ledger(ledger, output_dir)
    if strict and failures:
        raise RuntimeError(f"Semantic fidelity failed: {', '.join(failures)}")
    return ledger


def build_semantic_contract(model_path: Path) -> dict[str, Any]:
    """Build the stable v2 semantic contract for one model source."""
    system = GNNParsingSystem(strict_validation=False)
    parse_result = system.parse_file(model_path)
    if not parse_result.success or parse_result.model is None:
        errors = getattr(parse_result, "errors", []) or ["parse failed"]
        raise ValueError(f"Could not parse {model_path}: {'; '.join(errors)}")

    model = parse_result.model
    content = model_path.read_text(encoding="utf-8")
    schema_variables, variable_errors = parse_state_space(
        content, file_path=str(model_path)
    )
    known_variables = {variable.name for variable in schema_variables}
    _, connection_errors = parse_connections(
        content, known_variables=known_variables, file_path=str(model_path)
    )
    matrix_errors = validate_matrix_dimensions(
        content, schema_variables, file_path=str(model_path)
    )

    contract = {
        "schema": CONTRACT_SCHEMA_VERSION,
        "source_file": str(model_path),
        "model_identity": {
            "model_name": str(model.model_name),
            "version": str(model.version),
            "annotation": str(model.annotation),
        },
        "variables": _canonical_variables(model),
        "edges": _canonical_edges(model),
        "parameter_shapes": _canonical_parameter_shapes(model),
        "parameter_values": _canonical_parameter_values(model),
        "parameter_metadata": _canonical_parameter_metadata(model),
        "parameter_names": sorted(
            str(parameter.name) for parameter in model.parameters if parameter.name
        ),
        "equations": _canonical_equations(model),
        "time_specification": _canonical_time_specification(model),
        "ontology_mappings": _canonical_ontology_mappings(model),
        "declared_metadata": {
            "raw_sections": sorted(str(name) for name in model.raw_sections),
            "source_format": (
                model.source_format.value if model.source_format else "markdown"
            ),
            "extensions": sorted(str(name) for name in model.extensions),
        },
        "parser_diagnostics": {
            "variable_errors": [str(error) for error in variable_errors],
            "connection_errors": [str(error) for error in connection_errors],
            "matrix_errors": [str(error) for error in matrix_errors],
            "structure_issues": model.validate_structure(),
        },
    }
    contract["contract_hash"] = _contract_hash(contract)
    return contract


def compare_semantic_contracts(
    original: dict[str, Any], reparsed: dict[str, Any]
) -> list[SemanticDifference]:
    """Return strict differences between two semantic contracts."""
    differences: list[SemanticDifference] = []
    for field in (
        "model_identity",
        "variables",
        "edges",
        "parameter_shapes",
        "parameter_values",
        "parameter_metadata",
        "parameter_names",
        "equations",
        "time_specification",
        "ontology_mappings",
    ):
        if original.get(field) != reparsed.get(field):
            differences.append(
                SemanticDifference(
                    field=field,
                    message=f"{field} changed across semantic round trip",
                )
            )
    return differences


def _select_families(
    manifest_path: Path, family_names: Iterable[str] | None
) -> list[ModelFamily]:
    """Load the manifest and apply the shared family-selection rule."""
    return select_model_families(
        load_model_family_manifest(manifest_path), family_names
    )


def _run_family_fidelity(
    family: ModelFamily, formats: Sequence[str], output_dir: Path
) -> dict[str, Any]:
    model_results: list[dict[str, Any]] = []
    family_dir = output_dir / family.name
    family_dir.mkdir(parents=True, exist_ok=True)
    for relative_name in family.representative_files:
        source = family.target_dir / relative_name
        if not source.exists():
            raise FileNotFoundError(f"Representative fixture not found: {source}")
        model_results.append(_run_model_fidelity(source, family_dir, formats))

    failed_models = [
        result["source_file"]
        for result in model_results
        if result["status"] == "failed"
    ]
    return {
        "name": family.name,
        "description": family.description,
        "status": "failed" if failed_models else "passed",
        "model_count": len(model_results),
        "failed_models": failed_models,
        "models": model_results,
    }


def _run_model_fidelity(
    model_path: Path, family_dir: Path, formats: Sequence[str]
) -> dict[str, Any]:
    original_contract = build_semantic_contract(model_path)
    contract_path = family_dir / f"{model_path.stem}_semantic_contract.json"
    contract_path.write_text(
        json.dumps(original_contract, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    round_trips = [
        _run_format_round_trip(model_path, original_contract, family_dir, fmt)
        for fmt in formats
    ]
    failed_formats = [
        result["format"] for result in round_trips if result["status"] == "failed"
    ]
    return {
        "source_file": str(model_path),
        "contract_hash": original_contract["contract_hash"],
        "contract_artifact": str(contract_path),
        "status": "failed" if failed_formats else "passed",
        "round_trips": round_trips,
    }


def _run_format_round_trip(
    model_path: Path,
    original_contract: dict[str, Any],
    family_dir: Path,
    format_name: str,
) -> dict[str, Any]:
    system = GNNParsingSystem(strict_validation=False)
    source_result: Any = system.parse_file(model_path)
    if source_result.model is None:
        return {
            "format": format_name,
            "status": "failed",
            "reason": "source_parse_failed",
            "differences": [],
        }
    try:
        gnn_format = GNNFormat(format_name)
    except ValueError:
        return {
            "format": format_name,
            "status": "failed",
            "reason": "unsupported_format",
            "differences": [],
        }
    if gnn_format not in system.serializers or gnn_format not in system.parsers:
        return {
            "format": format_name,
            "status": "failed",
            "reason": "unsupported_serializer_or_parser",
            "differences": [],
        }

    try:
        serialized = system.serialize(source_result.model, gnn_format)
        artifact = family_dir / f"{model_path.stem}.{format_name}"
        artifact.write_text(serialized, encoding="utf-8")
        reparsed = system.parse_string(serialized, gnn_format)
        if not reparsed.success or reparsed.model is None:
            return {
                "format": format_name,
                "status": "failed",
                "reason": "reparse_failed",
                "artifact": str(artifact),
                "differences": [],
            }
        reparsed_contract = _contract_from_model(
            deepcopy(original_contract), reparsed.model
        )
        differences = compare_semantic_contracts(original_contract, reparsed_contract)
    except Exception as exc:  # pragma: no cover - defensive ledger detail
        return {
            "format": format_name,
            "status": "failed",
            "reason": f"round_trip_error: {exc}",
            "differences": [],
        }

    return {
        "format": format_name,
        "status": "failed" if differences else "passed",
        "reason": None if not differences else "semantic_contract_changed",
        "artifact": str(artifact),
        "differences": [difference.to_dict() for difference in differences],
    }


def _contract_from_model(source_contract: dict[str, Any], model: Any) -> dict[str, Any]:
    """Build a contract from an already parsed model using source diagnostics."""
    contract = deepcopy(source_contract)
    contract["source_file"] = "<serialized-round-trip>"
    contract["model_identity"] = {
        "model_name": str(model.model_name),
        "version": str(model.version),
        "annotation": str(model.annotation),
    }
    contract["variables"] = _canonical_variables(model)
    contract["edges"] = _canonical_edges(model)
    contract["parameter_shapes"] = _canonical_parameter_shapes(model)
    contract["parameter_values"] = _canonical_parameter_values(model)
    contract["parameter_metadata"] = _canonical_parameter_metadata(model)
    contract["parameter_names"] = sorted(
        str(parameter.name) for parameter in model.parameters if parameter.name
    )
    contract["equations"] = _canonical_equations(model)
    contract["time_specification"] = _canonical_time_specification(model)
    contract["ontology_mappings"] = _canonical_ontology_mappings(model)
    contract["declared_metadata"] = {
        "raw_sections": sorted(str(name) for name in model.raw_sections),
        "source_format": model.source_format.value if model.source_format else "json",
        "extensions": sorted(str(name) for name in model.extensions),
    }
    contract["contract_hash"] = _contract_hash(contract)
    return contract


def _canonical_variables(model: Any) -> list[dict[str, Any]]:
    variables = []
    for variable in model.variables:
        variables.append(
            {
                "name": str(variable.name),
                "dimensions": [
                    _normalize_dimension(dim) for dim in variable.dimensions
                ],
                "var_type": str(getattr(variable.var_type, "value", variable.var_type)),
                "data_type": str(
                    getattr(variable.data_type, "value", variable.data_type)
                ),
                "description": str(variable.description or ""),
                "constraints": _json_stable(variable.constraints),
            }
        )
    return sorted(variables, key=lambda item: item["name"])


def _canonical_edges(model: Any) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for connection in model.connections:
        connection_type = str(
            getattr(connection.connection_type, "value", connection.connection_type)
        )
        for source in connection.source_variables:
            for target in connection.target_variables:
                edges.append(
                    {
                        "source": str(source),
                        "target": str(target),
                        "connection_type": connection_type,
                        "weight": connection.weight,
                        "description": str(connection.description or ""),
                    }
                )
    return sorted(
        edges,
        key=lambda item: (
            item["source"],
            item["target"],
            item["connection_type"],
        ),
    )


def _canonical_parameter_shapes(model: Any) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for parameter in model.parameters:
        name = str(parameter.name).strip()
        if not name:
            continue
        shapes[name] = _shape_of(parameter.value)
    return dict(sorted(shapes.items()))


def _canonical_parameter_values(model: Any) -> dict[str, Any]:
    """Keep every parameter value and axis order in the scientific contract."""
    return dict(
        sorted(
            (str(parameter.name), _json_stable(parameter.value))
            for parameter in model.parameters
            if parameter.name
        )
    )


def _canonical_parameter_metadata(model: Any) -> dict[str, Any]:
    """Preserve declared parameter semantics and provenance across formats."""
    return dict(
        sorted(
            (str(parameter.name), _json_stable(parameter.metadata))
            for parameter in model.parameters
            if parameter.name and parameter.metadata
        )
    )


def _canonical_equations(model: Any) -> list[dict[str, str]]:
    equations = []
    for equation in getattr(model, "equations", []):
        equations.append(
            {
                "label": str(equation.label or ""),
                "content": str(equation.content),
                "format": str(equation.format),
                "description": str(equation.description or ""),
            }
        )
    return sorted(
        equations,
        key=lambda item: (
            item["label"],
            item["content"],
            item["format"],
            item["description"],
        ),
    )


def _canonical_time_specification(model: Any) -> dict[str, Any] | None:
    time_spec = getattr(model, "time_specification", None)
    if time_spec is None:
        return None
    return {
        "time_type": str(time_spec.time_type),
        "discretization": str(time_spec.discretization or ""),
        "horizon": time_spec.horizon,
        "step_size": time_spec.step_size,
    }


def _canonical_ontology_mappings(model: Any) -> list[dict[str, str]]:
    mappings = []
    for mapping in getattr(model, "ontology_mappings", []):
        mappings.append(
            {
                "variable_name": str(mapping.variable_name),
                "ontology_term": str(mapping.ontology_term),
                "description": str(mapping.description or ""),
            }
        )
    return sorted(
        mappings,
        key=lambda item: (
            item["variable_name"],
            item["ontology_term"],
            item["description"],
        ),
    )


def _shape_of(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        child_shape = _shape_of(value[0])
        return [len(value), *child_shape]
    return []


def _normalize_dimension(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return text


def _json_stable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _contract_hash(contract: dict[str, Any]) -> str:
    stable = {
        key: value
        for key, value in contract.items()
        if key not in {"contract_hash", "source_file", "parser_diagnostics"}
    }
    payload = json.dumps(stable, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_ledger(ledger: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "semantic_fidelity_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "semantic_fidelity_ledger.md").write_text(
        render_semantic_fidelity_markdown(ledger),
        encoding="utf-8",
    )
