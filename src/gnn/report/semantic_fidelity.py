"""Compatibility import for the semantic fidelity ledger renderer.

The renderer lives with its producer, :mod:`gnn.pipeline.semantic_fidelity`, so the
pipeline never imports :mod:`gnn.report`; this facade keeps the historical
``gnn.report.semantic_fidelity`` import path.
"""

from gnn.pipeline.semantic_fidelity import (
    render_semantic_fidelity_markdown as render_semantic_fidelity_markdown,
)
