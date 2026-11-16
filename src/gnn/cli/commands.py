"""Compatibility re-exports for the shared CLI/API command semantics.

The implementation moved below both transport adapters to
:mod:`gnn.operations.commands`; import from there in new code. This module
keeps every historical ``gnn.cli.commands`` name importable.
"""

from __future__ import annotations

from gnn.operations.commands import (
    EXTRACT_DEFECT_EMPTY_SKELETON as EXTRACT_DEFECT_EMPTY_SKELETON,
)
from gnn.operations.commands import (
    EXTRACT_DEFECT_ERROR_STATUS as EXTRACT_DEFECT_ERROR_STATUS,
)
from gnn.operations.commands import ValidationOutcome as ValidationOutcome
from gnn.operations.commands import build_parse_payload as build_parse_payload
from gnn.operations.commands import extract_payload_defect as extract_payload_defect
from gnn.operations.commands import find_render_artifact as find_render_artifact
from gnn.operations.commands import preflight_severities as preflight_severities
from gnn.operations.commands import (
    publish_pipeline_report as publish_pipeline_report,
)
from gnn.operations.commands import (
    render_processing_succeeded as render_processing_succeeded,
)
from gnn.operations.commands import run_validation_checks as run_validation_checks

__all__ = [
    "EXTRACT_DEFECT_EMPTY_SKELETON",
    "EXTRACT_DEFECT_ERROR_STATUS",
    "ValidationOutcome",
    "build_parse_payload",
    "extract_payload_defect",
    "find_render_artifact",
    "preflight_severities",
    "publish_pipeline_report",
    "run_validation_checks",
    "render_processing_succeeded",
]
