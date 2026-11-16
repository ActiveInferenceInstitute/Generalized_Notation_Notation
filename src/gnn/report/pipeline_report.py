#!/usr/bin/env python3
"""Compatibility import for the ``PIPELINE_REPORT.md`` generator.

The generator lives with its producer, :mod:`gnn.pipeline.summary_wiring`, which
writes the final report at pipeline finalization; keeping it there means the
pipeline never imports :mod:`gnn.report`. This facade preserves the historical
``gnn.report.pipeline_report`` import path and section helpers.
"""

from gnn.pipeline.summary_wiring import PipelineReportContext as PipelineReportContext
from gnn.pipeline.summary_wiring import _build_report_context as _build_report_context
from gnn.pipeline.summary_wiring import _collect_step_stats as _collect_step_stats
from gnn.pipeline.summary_wiring import _discover_step_dirs as _discover_step_dirs
from gnn.pipeline.summary_wiring import _load_summary as _load_summary
from gnn.pipeline.summary_wiring import _section_artifacts as _section_artifacts
from gnn.pipeline.summary_wiring import _section_errors as _section_errors
from gnn.pipeline.summary_wiring import _section_header as _section_header
from gnn.pipeline.summary_wiring import _section_statistics as _section_statistics
from gnn.pipeline.summary_wiring import _section_step_status as _section_step_status
from gnn.pipeline.summary_wiring import _section_timing as _section_timing
from gnn.pipeline.summary_wiring import (
    generate_pipeline_report as generate_pipeline_report,
)

__all__ = ["PipelineReportContext", "generate_pipeline_report"]
