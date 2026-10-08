#!/usr/bin/env python3
"""
Stateless JSON-schema parameter validation for MCP tools.

Extracted from ``mcp.py`` (STR-2 phase 1): these helpers inspect only their
arguments — no registry, executor, or instance state — so they live here as
module-level functions. ``MCP._validate_params`` remains a thin delegating
wrapper, keeping all call sites and error behavior identical to the
pre-extraction implementation.
"""

import math
import re
from typing import Any, Dict

from .exceptions import MCPValidationError


def _validate_params(
    schema: Dict[str, Any], params: Dict[str, Any], *, strict: bool
) -> None:
    """
    Enhanced parameter validation against schema with detailed error reporting.

    Args:
        schema: JSON schema for validation
        params: Parameters to validate
        strict: When False, only required-parameter presence is checked;
            when True, full per-field schema validation runs

    Raises:
        MCPValidationError: If validation fails
    """
    if not isinstance(params, dict):
        raise MCPValidationError("Parameters must be a dictionary")
    if not strict:
        # Basic validation only
        if "required" in schema:
            for required in schema["required"]:
                if required not in params:
                    raise MCPValidationError(f"Missing required parameter: {required}")
        return

    # Check required fields
    if "required" in schema:
        for required_field in schema["required"]:
            if required_field not in params:
                raise MCPValidationError(
                    f"Required parameter '{required_field}' is missing",
                    field=required_field,
                )

    # Validate properties
    _validate_field("parameters", params, {**schema, "type": "object"})

    # Validate additional constraints
    if "minProperties" in schema and len(params) < schema["minProperties"]:
        raise MCPValidationError(
            f"Too few properties: {len(params)} < {schema['minProperties']}"
        )

    if "maxProperties" in schema and len(params) > schema["maxProperties"]:
        raise MCPValidationError(
            f"Too many properties: {len(params)} > {schema['maxProperties']}"
        )


def _validate_field(
    field_name: str, field_value: Any, field_schema: Dict[str, Any]
) -> None:
    """
    Validate a single field against its schema.

    Args:
        field_name: Name of the field
        field_value: Value to validate
        field_schema: Schema for the field

    Raises:
        MCPValidationError: If validation fails
    """
    field_type = field_schema.get("type")

    # Type validation
    if field_type == "string":
        if not isinstance(field_value, str):
            raise MCPValidationError(
                f"Parameter '{field_name}' must be a string",
                field=field_name,
                value=field_value,
            )

        # String-specific validations
        if "minLength" in field_schema and len(field_value) < field_schema["minLength"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too short: {len(field_value)} < {field_schema['minLength']}",
                field=field_name,
                value=field_value,
            )

        if "maxLength" in field_schema and len(field_value) > field_schema["maxLength"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too long: {len(field_value)} > {field_schema['maxLength']}",
                field=field_name,
                value=field_value,
            )

        if "pattern" in field_schema:
            if not re.match(field_schema["pattern"], field_value):
                raise MCPValidationError(
                    f"Parameter '{field_name}' does not match pattern: {field_schema['pattern']}",
                    field=field_name,
                    value=field_value,
                )

        if "enum" in field_schema and field_value not in field_schema["enum"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' must be one of: {field_schema['enum']}",
                field=field_name,
                value=field_value,
            )

    elif field_type == "integer":
        if isinstance(field_value, bool) or not isinstance(field_value, int):
            raise MCPValidationError(
                f"Parameter '{field_name}' must be an integer",
                field=field_name,
                value=field_value,
            )

        # Integer-specific validations
        if "minimum" in field_schema and field_value < field_schema["minimum"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too small: {field_value} < {field_schema['minimum']}",
                field=field_name,
                value=field_value,
            )

        if "maximum" in field_schema and field_value > field_schema["maximum"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too large: {field_value} > {field_schema['maximum']}",
                field=field_name,
                value=field_value,
            )

    elif field_type == "number":
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(field_value)
        ):
            raise MCPValidationError(
                f"Parameter '{field_name}' must be a number",
                field=field_name,
                value=field_value,
            )

        # Number-specific validations
        if "minimum" in field_schema and field_value < field_schema["minimum"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too small: {field_value} < {field_schema['minimum']}",
                field=field_name,
                value=field_value,
            )

        if "maximum" in field_schema and field_value > field_schema["maximum"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too large: {field_value} > {field_schema['maximum']}",
                field=field_name,
                value=field_value,
            )

    elif field_type == "boolean":
        if not isinstance(field_value, bool):
            raise MCPValidationError(
                f"Parameter '{field_name}' must be a boolean",
                field=field_name,
                value=field_value,
            )

    elif field_type == "array":
        if not isinstance(field_value, list):
            raise MCPValidationError(
                f"Parameter '{field_name}' must be an array",
                field=field_name,
                value=field_value,
            )

        # Array-specific validations
        if "minItems" in field_schema and len(field_value) < field_schema["minItems"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too few items: {len(field_value)} < {field_schema['minItems']}",
                field=field_name,
                value=field_value,
            )

        if "maxItems" in field_schema and len(field_value) > field_schema["maxItems"]:
            raise MCPValidationError(
                f"Parameter '{field_name}' too many items: {len(field_value)} > {field_schema['maxItems']}",
                field=field_name,
                value=field_value,
            )

        # Validate array items if schema provided
        if "items" in field_schema:
            for i, item in enumerate(field_value):
                try:
                    _validate_field(f"{field_name}[{i}]", item, field_schema["items"])
                except MCPValidationError as e:
                    raise MCPValidationError(
                        f"Array item validation failed: {e}",
                        field=field_name,
                        value=field_value,
                    ) from e

    elif field_type == "object":
        if not isinstance(field_value, dict):
            raise MCPValidationError(
                f"Parameter '{field_name}' must be an object",
                field=field_name,
                value=field_value,
            )

        for required in field_schema.get("required", []):
            if required not in field_value:
                raise MCPValidationError(
                    f"Required parameter '{field_name}.{required}' is missing",
                    field=f"{field_name}.{required}",
                )

        # Object-specific validations
        if "properties" in field_schema:
            for prop_name, prop_value in field_value.items():
                if prop_name in field_schema["properties"]:
                    try:
                        _validate_field(
                            f"{field_name}.{prop_name}",
                            prop_value,
                            field_schema["properties"][prop_name],
                        )
                    except MCPValidationError as e:
                        raise MCPValidationError(
                            f"Object property validation failed: {e}",
                            field=field_name,
                            value=field_value,
                        ) from e

        # Check for additional properties
        if (
            "additionalProperties" in field_schema
            and field_schema["additionalProperties"] is False
        ):
            allowed_props = set(field_schema.get("properties", {}).keys())
            actual_props = set(field_value.keys())
            extra_props = actual_props - allowed_props
            if extra_props:
                raise MCPValidationError(
                    f"Parameter '{field_name}' has additional properties not allowed: {extra_props}",
                    field=field_name,
                    value=field_value,
                )
