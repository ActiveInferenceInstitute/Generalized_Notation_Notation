"""
Semantic Validator

This module provides semantic validation for GNN models, including
structure validation, type checking, and consistency verification.
"""

import logging
import re
from pathlib import Path
from typing import Any

from .structure import (
    DirectedEdge,
    clamp01,
    cycle_nodes,
    display_file_name,
    extract_content_from_dict,
)

logger = logging.getLogger(__name__)


class SemanticValidator:
    """Validator for semantic aspects of GNN models."""

    def __init__(self, validation_level: str = "standard") -> None:
        """
        Initialize the semantic validator.

        Args:
            validation_level: Validation level (basic, standard, strict, research)
        """
        self.validation_level = validation_level
        self.validation_rules = self._get_validation_rules()

    def validate(self, content: str) -> dict[str, Any]:
        """
        Validate the semantic aspects of a GNN model.

        Args:
            content: GNN model content

        Returns:
            Validation result with errors and warnings
        """
        errors: list[Any] = []
        warnings: list[Any] = []

        # Canonical markdown and parsed-model receipts share the same reference
        # checks. Documents using the earlier block-section syntax continue
        # through the existing rule set.
        if re.search(r"^\s*## ", content, re.MULTILINE):
            from gnn.parsers.markdown_parser import MarkdownGNNParser

            parsed = MarkdownGNNParser().parse_string(content)
            errors.extend(parsed.errors)
            model = parsed.model
            names = {variable.name for variable in model.variables}
            # Inspect raw dimension tokens before the parser's intentional
            # symbolic-dimension normalization can turn negative integers into 1.
            state_space = model.raw_sections.get("StateSpaceBlock", "")
            for raw_line in state_space.splitlines():
                declaration = raw_line.split("#", 1)[0].strip()
                match = re.match(r"([^\[\]]+)\[([^\]]*)\]", declaration)
                if match is None:
                    continue
                name, dimensions = match.groups()
                for dimension in dimensions.split(","):
                    token = dimension.strip()
                    if re.fullmatch(r"[+-]?\d+", token) and int(token) <= 0:
                        errors.append(
                            f"Variable '{name.strip()}' has non-positive dimensions"
                        )
                        break
            if self._get_level_value() >= 2:
                for connection in model.connections:
                    for name in (
                        *connection.source_variables,
                        *connection.target_variables,
                    ):
                        if name not in names:
                            errors.append(
                                f"Connection references undeclared variable '{name}'"
                            )

        # Apply validation rules based on level
        for rule in self.validation_rules:
            if rule["level"] <= self._get_level_value():
                result = rule["validator"](content)
                errors.extend(result.get("errors", []))
                warnings.extend(result.get("warnings", []))

        return {
            "is_valid": len(errors) == 0,
            "validation_level": self.validation_level,
            "errors": errors,
            "warnings": warnings,
        }

    def _get_level_value(self) -> int:
        """Convert validation level string to numeric value."""
        levels: dict[str, int] = {"basic": 1, "standard": 2, "strict": 3, "research": 4}
        # Default to standard
        return levels.get(self.validation_level.lower(), 2)

    def _get_validation_rules(self) -> list[dict[str, Any]]:
        """Get validation rules based on validation level."""
        return [
            {
                "name": "basic_structure",
                "level": 1,  # Basic
                "validator": self._validate_basic_structure,
            },
            {
                "name": "state_space_definitions",
                "level": 1,  # Basic
                "validator": self._validate_state_space_definitions,
            },
            {
                "name": "connection_integrity",
                "level": 2,  # Standard
                "validator": self._validate_connection_integrity,
            },
            {
                "name": "mathematical_consistency",
                "level": 2,  # Standard
                "validator": self._validate_mathematical_consistency,
            },
            {
                "name": "active_inference_principles",
                "level": 3,  # Strict
                "validator": self._validate_active_inference_principles,
            },
            {
                "name": "causal_relationships",
                "level": 3,  # Strict
                "validator": self._validate_causal_relationships,
            },
            {
                "name": "advanced_mathematical_properties",
                "level": 4,  # Research
                "validator": self._validate_advanced_mathematical_properties,
            },
        ]

    def _validate_basic_structure(self, content: str) -> dict[str, Any]:
        """Validate basic structure of the GNN model."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Check for required elements
        if not re.search(r"(?:ModelName:|^\s*## ModelName\s*$)", content, re.MULTILINE):
            warnings.append("Missing ModelName definition")

        if not re.search(r"StateSpaceBlock", content):
            errors.append("Missing StateSpaceBlock definition")

        if not re.search(r"Connection", content) and re.search(
            r"StateSpaceBlock", content
        ):
            warnings.append("Model has StateSpaceBlocks but no Connections")

        # Check for proper block structure
        state_blocks = re.findall(r"StateSpaceBlock\s*\{([^}]*)\}", content)
        for i, block in enumerate(state_blocks):
            if not block.strip():
                errors.append(f"Empty StateSpaceBlock at index {i}")

        return {"errors": errors, "warnings": warnings}

    def _validate_state_space_definitions(self, content: str) -> dict[str, Any]:
        """Validate state space definitions."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Extract state blocks
        state_blocks = re.findall(r"StateSpaceBlock\s*\{([^}]*)\}", content)

        for i, block in enumerate(state_blocks):
            # Check for required fields
            if not re.search(r"Name:", block):
                errors.append(f"StateSpaceBlock at index {i} missing Name field")

            if not re.search(r"Dimensions:", block):
                errors.append(f"StateSpaceBlock at index {i} missing Dimensions field")

            # Check dimensions format
            dim_match = re.search(r"Dimensions:\s*([^\n]+)", block)
            if dim_match:
                dims_str = dim_match.group(1).strip()
                try:
                    dims = [int(d.strip()) for d in dims_str.split(",")]
                    if any(d <= 0 for d in dims):
                        errors.append(
                            f"StateSpaceBlock at index {i} has non-positive dimensions: {dims_str}"
                        )
                except ValueError:
                    errors.append(
                        f"StateSpaceBlock at index {i} has invalid dimensions format: {dims_str}"
                    )

        return {"errors": errors, "warnings": warnings}

    def _validate_connection_integrity(self, content: str) -> dict[str, Any]:
        """Validate connection integrity."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Extract state blocks and connections
        state_blocks = re.findall(r"StateSpaceBlock\s*\{([^}]*)\}", content)
        connections = re.findall(r"Connection\s*\{([^}]*)\}", content)

        # Extract block names
        block_names: list[Any] = []
        for block in state_blocks:
            name_match = re.search(r"Name:\s*([^\n]+)", block)
            if name_match:
                block_names.append(name_match.group(1).strip())

        # Check connections
        for i, conn in enumerate(connections):
            # Check for required fields
            if not re.search(r"From:", conn):
                errors.append(f"Connection at index {i} missing From field")

            if not re.search(r"To:", conn):
                errors.append(f"Connection at index {i} missing To field")

            # Check if referenced blocks exist
            from_match = re.search(r"From:\s*([^\n]+)", conn)
            to_match = re.search(r"To:\s*([^\n]+)", conn)

            if from_match and from_match.group(1).strip() not in block_names:
                errors.append(
                    f"Connection {i} references non-existent 'From' block: {from_match.group(1).strip()}"
                )

            if to_match and to_match.group(1).strip() not in block_names:
                errors.append(
                    f"Connection {i} references non-existent 'To' block: {to_match.group(1).strip()}"
                )

        # Check for isolated blocks
        connected_blocks: set[Any] = set()
        for conn in connections:
            from_match = re.search(r"From:\s*([^\n]+)", conn)
            to_match = re.search(r"To:\s*([^\n]+)", conn)

            if from_match:
                connected_blocks.add(from_match.group(1).strip())
            if to_match:
                connected_blocks.add(to_match.group(1).strip())

        isolated_blocks = [name for name in block_names if name not in connected_blocks]
        if isolated_blocks:
            warnings.append(
                f"Isolated blocks with no connections: {', '.join(isolated_blocks)}"
            )

        return {"errors": errors, "warnings": warnings}

    def _validate_mathematical_consistency(self, content: str) -> dict[str, Any]:
        """Validate mathematical consistency."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Extract state blocks and connections
        state_blocks = re.findall(r"StateSpaceBlock\s*\{([^}]*)\}", content)
        connections = re.findall(r"Connection\s*\{([^}]*)\}", content)

        # Extract block dimensions
        block_dims: dict[Any, Any] = {}
        for block in state_blocks:
            name_match = re.search(r"Name:\s*([^\n]+)", block)
            dim_match = re.search(r"Dimensions:\s*([^\n]+)", block)

            if name_match and dim_match:
                name = name_match.group(1).strip()
                try:
                    dims = [
                        int(d.strip()) for d in dim_match.group(1).strip().split(",")
                    ]
                    block_dims[name] = dims
                except ValueError:
                    logger.debug(
                        "Non-integer dimensions for '%s', skipping compatibility check",
                        name,
                    )

        # Check connection dimension compatibility
        for i, conn in enumerate(connections):
            from_match = re.search(r"From:\s*([^\n]+)", conn)
            to_match = re.search(r"To:\s*([^\n]+)", conn)
            mapping_match = re.search(r"Mapping:\s*([^\n]+)", conn)

            if from_match and to_match:
                from_block = from_match.group(1).strip()
                to_block = to_match.group(1).strip()

                if from_block in block_dims and to_block in block_dims:
                    from_dims = block_dims[from_block]
                    to_dims = block_dims[to_block]

                    # Check if mapping is specified
                    if mapping_match:
                        mapping = mapping_match.group(1).strip()
                        # Validate mapping specification
                        valid_mappings: list[Any] = [
                            "identity",
                            "transpose",
                            "reshape",
                            "broadcast",
                            "reduce",
                        ]
                        mapping_lower = mapping.lower()

                        if mapping_lower == "identity":
                            # Identity mapping: dimensions must match exactly
                            if from_dims != to_dims:
                                errors.append(
                                    f"Connection {i} uses 'identity' mapping but dimensions don't match: {from_dims} -> {to_dims}"
                                )
                        elif mapping_lower == "transpose":
                            # Transpose mapping: must be 2D and dimensions reversed
                            if len(from_dims) != 2 or len(to_dims) != 2:
                                errors.append(
                                    f"Connection {i} uses 'transpose' mapping on non-2D tensors"
                                )
                            elif from_dims != list(reversed(to_dims)):
                                errors.append(
                                    f"Connection {i} uses 'transpose' but dimensions are not reversed: {from_dims} -> {to_dims}"
                                )
                        elif mapping_lower == "reshape":
                            # Reshape mapping: total elements must match
                            from_total = 1
                            to_total = 1
                            for d in from_dims:
                                from_total *= d
                            for d in to_dims:
                                to_total *= d
                            if from_total != to_total:
                                errors.append(
                                    f"Connection {i} uses 'reshape' but total elements differ: {from_total} vs {to_total}"
                                )
                        elif mapping_lower == "broadcast":
                            # Broadcast mapping: target dimensions must be >= source
                            for j, (from_dim, to_dim) in enumerate(
                                zip(from_dims, to_dims)
                            ):
                                if to_dim < from_dim:
                                    warnings.append(
                                        f"Connection {i} uses 'broadcast' but dimension {j} shrinks: {from_dim} -> {to_dim}"
                                    )
                        elif mapping_lower == "reduce":
                            # Reduce mapping: source dimensions can be larger.
                            continue
                        elif mapping_lower not in valid_mappings:
                            warnings.append(
                                f"Connection {i} uses unknown mapping type: '{mapping}' (supported: {valid_mappings})"
                            )
                    else:
                        # Default mapping: check if dimensions match
                        if len(from_dims) != len(to_dims):
                            warnings.append(
                                f"Connection {i} between blocks with different dimension counts: {from_block}({len(from_dims)}) -> {to_block}({len(to_dims)})"
                            )

                        # Check if dimensions are compatible
                        for j, (from_dim, to_dim) in enumerate(zip(from_dims, to_dims)):
                            if from_dim != to_dim:
                                warnings.append(
                                    f"Connection {i} has mismatched dimensions at index {j}: {from_dim} -> {to_dim}"
                                )

        return {"errors": errors, "warnings": warnings}

    def _validate_active_inference_principles(self, content: str) -> dict[str, Any]:
        """Validate compliance with Active Inference principles."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Check for required components in Active Inference models
        has_observation = bool(
            re.search(r"ObservationModel|Observation", content, re.IGNORECASE)
        )
        has_transition = bool(
            re.search(r"TransitionModel|Dynamics", content, re.IGNORECASE)
        )
        has_prior = bool(re.search(r"Prior|PriorPreferences", content, re.IGNORECASE))

        if not has_observation:
            warnings.append("Active Inference model missing explicit observation model")

        if not has_transition:
            warnings.append(
                "Active Inference model missing explicit transition dynamics"
            )

        if not has_prior:
            warnings.append(
                "Active Inference model missing explicit priors or preferences"
            )

        # Check for free energy minimization principle
        has_free_energy = bool(
            re.search(r"FreeEnergy|VariationalFreeEnergy|VFE", content, re.IGNORECASE)
        )
        if not has_free_energy:
            warnings.append(
                "Active Inference model does not explicitly mention free energy minimization"
            )

        return {"errors": errors, "warnings": warnings}

    def _validate_causal_relationships(self, content: str) -> dict[str, Any]:
        """Validate causal relationships in the model."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Extract connections
        connections = re.findall(r"Connection\s*\{([^}]*)\}", content)

        # Check for circular dependencies with exact cycle membership
        endpoints: list[str] = []
        seen_endpoints: set[str] = set()
        edges: list[DirectedEdge] = []
        for conn in connections:
            from_match = re.search(r"From:\s*([^\n]+)", conn)
            to_match = re.search(r"To:\s*([^\n]+)", conn)

            if from_match and to_match:
                from_block = from_match.group(1).strip()
                to_block = to_match.group(1).strip()
                for endpoint in (from_block, to_block):
                    if endpoint not in seen_endpoints:
                        seen_endpoints.add(endpoint)
                        endpoints.append(endpoint)
                edges.append(DirectedEdge(source=from_block, target=to_block))

        cycles = cycle_nodes(endpoints, edges)
        if cycles:
            warnings.append(
                f"Potential circular dependencies detected in blocks: {', '.join(cycles)}"
            )

        return {"errors": errors, "warnings": warnings}

    def _validate_advanced_mathematical_properties(
        self, content: str
    ) -> dict[str, Any]:
        """Validate advanced mathematical properties."""
        errors: list[Any] = []
        warnings: list[Any] = []

        # Research-level validation checks for explicit advanced concepts.
        has_markov_blanket = bool(re.search(r"MarkovBlanket", content, re.IGNORECASE))
        has_information_geometry = bool(
            re.search(r"InformationGeometry|FisherInformation", content, re.IGNORECASE)
        )
        has_ergodicity = bool(re.search(r"Ergodic|Ergodicity", content, re.IGNORECASE))

        if not has_markov_blanket:
            warnings.append("Advanced model does not explicitly define Markov blankets")

        if not has_information_geometry:
            warnings.append(
                "Advanced model does not reference information geometry concepts"
            )

        if not has_ergodicity:
            warnings.append("Advanced model does not address ergodicity assumptions")

        return {"errors": errors, "warnings": warnings}


def process_semantic_validation(
    model_data: str | Path | dict[str, Any], **kwargs: Any
) -> dict[str, Any]:
    """
    Process semantic validation for a GNN model.

    Args:
        model_data: Path to GNN file, Path object, or model data dictionary
        **kwargs: Additional arguments including validation_level

    Returns:
        Dictionary with validation results
    """

    file_path = "unknown"
    try:
        if isinstance(model_data, dict):
            content = extract_content_from_dict(model_data)
            file_path = str(model_data.get("file_path", "unknown"))
            if not model_data.get("raw_sections"):
                content = re.sub(
                    r"^(StateSpaceBlock|Connections):",
                    r"## \1",
                    content,
                    flags=re.MULTILINE,
                )
        else:
            file_path = str(model_data)
            content = Path(model_data).read_text(encoding="utf-8")
        validate_probabilities = bool(kwargs.get("strict_probabilities", False))
        if (
            validate_probabilities
            and isinstance(model_data, dict)
            and not model_data.get("raw_sections")
        ):
            content = _with_structured_probabilities(content, model_data)
        result = validate_content(
            content,
            validation_level=kwargs.get("validation_level", "standard"),
            strict_probabilities=validate_probabilities,
        )
        if isinstance(model_data, dict) and "probability_validation" in result:
            result["probability_validation"]["line_origin"] = "reconstructed_model"
        result.update(file_path=file_path, file_name=display_file_name(file_path))
        return result
    except Exception as exc:
        return {
            "status": "error",
            "file_path": file_path,
            "file_name": display_file_name(file_path),
            "error": str(exc),
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "semantic_score": 0.0,
            "recovery": True,
        }


def _with_structured_probabilities(content: str, model: dict[str, Any]) -> str:
    """Expose numeric declarations carried without raw source section text."""
    from gnn.extract.pomdp_support import find_section_heading

    if find_section_heading(content, "InitialParameterization") is not None:
        return content
    values = model.get("initialparameterization") or model.get(
        "initial_parameterization"
    )
    if not values:
        parameters = model.get("parameters")
        if isinstance(parameters, dict):
            values = parameters
        elif isinstance(parameters, list):
            values = {
                item["name"]: item["value"]
                for item in parameters
                if isinstance(item, dict)
                and isinstance(item.get("name"), str)
                and "value" in item
            }
    if not isinstance(values, dict):
        return content
    matrices = [
        f"{key} = {value!r}"
        for key, value in values.items()
        if isinstance(key, str)
        and re.fullmatch(r"(?:[ABCD](?:_\w+)?|[FHQR]|prior_mean|prior_cov)", key)
    ]
    if not matrices:
        return content
    parameters = dict(model.get("model_parameters") or {})
    options = {
        key: parameters.get(key, values.get(key))
        for key in (
            "num_hidden_states",
            "num_states",
            "num_obs",
            "num_observations",
            "num_actions",
        )
    }
    for key in parameters.keys() | values.keys():
        if isinstance(key, str) and re.fullmatch(
            r"(?:[bB]_tensor_order|transition_tensor_order)(?:_\w+)?", key
        ):
            order = parameters.get(key, values.get(key))
            if isinstance(order, str) and re.fullmatch(r"[\w -]+", order):
                options[key] = order
    return (
        content
        + "\n\n## InitialParameterization\n"
        + "\n".join(matrices)
        + "\n\n## ModelParameters\n"
        + "\n".join(
            f"{key}: {value}"
            for key, value in options.items()
            if type(value) is int
            or isinstance(value, str)
            and re.fullmatch(r"[\w -]+", value)
        )
    )


def _probability_receipt(content: str) -> dict[str, Any]:
    """Evaluate the same authored categorical contract as strict extraction."""
    from gnn.extract.pomdp_extractor import extract_pomdp_from_content
    from gnn.extract.pomdp_stochastic import DEFAULT_ATOL
    from gnn.extract.pomdp_support import find_section_heading

    data: dict[str, Any] = {
        "contract": "gnn_strict_categorical_probabilities",
        "absolute_tolerance": DEFAULT_ATOL,
        "status": "not_applicable",
        "diagnostics": [],
        "line_origin": "source_text",
    }
    if find_section_heading(content, "InitialParameterization") is None:
        return data
    space, errors = extract_pomdp_from_content(
        content,
        on_error="collect",
        strict_probabilities=True,
        insert_default_c=False,
    )
    if errors:
        data["status"] = "failed"
        data["diagnostics"] = [
            {
                "code": error.code,
                "message": error.message,
                "line": error.line,
                "section": error.section,
            }
            for error in errors
        ]
    elif (
        space is not None
        and space.model_kind == "discrete"
        and any(re.fullmatch(r"[ABCD](?:_\w+)?", key) for key in (space.matrices or {}))
    ):
        data["status"] = "passed"
    return data


def validate_content(
    content: str,
    validation_level: str = "standard",
    *,
    strict_probabilities: bool = False,
) -> dict[str, Any]:
    """
    Validate raw GNN content text without any file I/O.

    Companion to ``process_semantic_validation`` for callers that already
    hold the model text in memory (MCP tools, editors, pipelines). Returns
    the same receipt shape with ``file_path``/``file_name`` set to
    "unknown" and never raises.
    """
    try:
        validator = SemanticValidator(validation_level)
        validation_result = validator.validate(content)
        probability = _probability_receipt(content) if strict_probabilities else None
        if probability and probability["status"] == "failed":
            validation_result["errors"].extend(
                f"{row['code']}: {row['message']}" for row in probability["diagnostics"]
            )
            validation_result["is_valid"] = False
        result = {
            "file_path": "unknown",
            "file_name": "unknown",
            "valid": validation_result["is_valid"],
            "errors": validation_result["errors"],
            "warnings": validation_result["warnings"],
            "semantic_score": _calculate_semantic_score(validation_result),
            "recovery": False,
        }
        if probability is not None:
            result["probability_validation"] = probability
        return result
    except Exception as e:
        return {
            "status": "error",
            "file_path": "unknown",
            "file_name": "unknown",
            "error": str(e),
            "valid": False,
            "errors": [str(e)],
            "warnings": [],
            "semantic_score": 0.0,
            "recovery": True,
        }


def _calculate_semantic_score(validation_result: dict[str, Any]) -> float:
    """Calculate a semantic score from validation results."""
    # Base score starts at 1.0
    score = 1.0

    # Deduct points for errors
    errors = validation_result.get("errors", [])
    score -= len(errors) * 0.2

    # Deduct points for warnings
    warnings = validation_result.get("warnings", [])
    score -= len(warnings) * 0.05

    # Ensure score is between 0 and 1
    return clamp01(score)
