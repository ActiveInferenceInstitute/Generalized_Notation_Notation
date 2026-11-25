"""Parse optional declarations at the boundary of one GNN model."""

import re
from collections.abc import Iterable

from .common import normalize_variable_name

_FORMS = {"?", "point", "range", "interval", "distribution", "samples"}
_FIELDS = {"unit", "convention", "support", "form"}
_FIELD_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*(?:\.[A-Za-z][A-Za-z0-9_-]*)*")


class BoundaryAnnotationError(ValueError):
    """The boundary section is present but invalid."""


def parse_boundary_annotation(content: str) -> dict[str, dict[str, str]]:
    """Read one declaration per variable from a Markdown section."""
    declarations: dict[str, dict[str, str]] = {}
    for line in content.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        label, colon, pairs = line.partition(":")
        variable = label.strip()
        if not colon or not variable:
            raise BoundaryAnnotationError(
                f"Expected a boundary declaration for a variable: {line!r}"
            )
        if variable in declarations:
            raise BoundaryAnnotationError(
                f"Duplicate boundary declaration for {variable!r}"
            )
        fields: dict[str, str] = {}
        for pair in pairs.split(";"):
            key, equals, value = pair.strip().partition("=")
            if not equals:
                raise BoundaryAnnotationError(
                    f"Expected key=value in the declaration for {variable!r}"
                )
            key = key.strip()
            if key in fields:
                raise BoundaryAnnotationError(
                    f"Duplicate boundary field {key!r} for {variable!r}"
                )
            fields[key] = value.strip()
        declarations[variable] = fields
    if not declarations:
        raise BoundaryAnnotationError(
            "BoundaryAnnotation must contain at least one declaration"
        )
    return declarations


def read_boundary_annotation(
    value: object, declared: set[str]
) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Validate structured declarations and retain extension fields."""
    if not isinstance(value, dict) or not value:
        raise BoundaryAnnotationError(
            "BoundaryAnnotation must map variables to declaration fields"
        )
    annotations: dict[str, dict[str, str]] = {}
    warnings: list[str] = []
    for name, fields in value.items():
        if not isinstance(name, str):
            raise BoundaryAnnotationError(
                f"BoundaryAnnotation references unknown variable {name!r}"
            )
        variable = _resolve_variable(name, declared)
        if variable in annotations:
            raise BoundaryAnnotationError(
                f"Duplicate boundary declaration for {variable!r}"
            )
        if not isinstance(fields, dict) or not fields:
            raise BoundaryAnnotationError(f"No boundary fields for {variable!r}")
        annotations[variable] = {}
        for key, text in fields.items():
            if not isinstance(key, str) or not _FIELD_NAME.fullmatch(key):
                raise BoundaryAnnotationError(
                    f"Invalid boundary field name {key!r} for {variable!r}"
                )
            if (
                not isinstance(text, str)
                or not text.strip()
                or any(c in text for c in ";\r\n")
            ):
                raise BoundaryAnnotationError(
                    f"Invalid text for boundary field {key!r} on {variable!r}"
                )
            text = text.strip()
            if key == "form" and text not in _FORMS:
                raise BoundaryAnnotationError(
                    f"Unsupported boundary value form {text!r} for {variable!r}"
                )
            if key not in _FIELDS and not key.startswith("convention."):
                warnings.append(
                    f"Unknown BoundaryAnnotation field {key!r} for {variable!r} was preserved"
                )
            annotations[variable][key] = text
    return annotations, warnings


def _resolve_variable(name: str, declared: set[str]) -> str:
    if name in declared:
        return name
    canonical = normalize_variable_name(name)
    aliases = [
        variable
        for variable in declared
        if normalize_variable_name(variable) == canonical
    ]
    if len(aliases) == 1:
        return aliases[0]
    if aliases:
        raise BoundaryAnnotationError(
            f"BoundaryAnnotation has an ambiguous variable reference {name!r}"
        )
    raise BoundaryAnnotationError(
        f"BoundaryAnnotation references unknown variable {name!r}"
    )


def reject_boundary_duplicates(
    declarations: Iterable[tuple[str, object]],
) -> None:
    """Reject duplicate keys before key/value pairs collapse into mappings."""
    variables: set[str] = set()
    for variable, fields in declarations:
        if variable in variables:
            raise BoundaryAnnotationError(
                f"Duplicate boundary declaration for {variable!r}"
            )
        variables.add(variable)
        keys: set[str] = set()
        if not isinstance(fields, list):
            continue
        for field in fields:
            if not isinstance(field, (tuple, list)) or len(field) != 2:
                continue
            key, _ = field
            if not isinstance(key, str):
                continue
            if key in keys:
                raise BoundaryAnnotationError(
                    f"Duplicate boundary field {key!r} for {variable!r}"
                )
            keys.add(key)
