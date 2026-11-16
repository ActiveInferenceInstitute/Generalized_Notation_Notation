"""Compatibility import for the cross-framework reliability ledger renderer.

The renderer lives with its producer, :mod:`gnn.pipeline.cross_framework_reliability`, so the
pipeline never imports :mod:`gnn.report`; this facade keeps the historical
``gnn.report.cross_framework_reliability`` import path.
"""

from gnn.pipeline.cross_framework_reliability import (
    render_cross_framework_reliability_markdown as render_cross_framework_reliability_markdown,
)
