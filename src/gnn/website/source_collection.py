"""Verified source discovery and model-page data, separate from output census."""

import logging
from pathlib import Path
from typing import Any

from gnn.parsers.common import ParseError

logger = logging.getLogger(__name__)


def _collect_gnn_files(p_root: Path, input_dir: Path) -> tuple[list[Path], bool]:
    """Discover GNN source markdown files, preferring ``<root>/input/gnn_files``."""
    from gnn.processing.discovery import is_model_source_path

    for search_dir in (input_dir,):
        if search_dir.exists():
            return sorted(
                p
                for p in search_dir.rglob("*.md")
                if is_model_source_path(p) and "archived_gnn_files" not in p.parts
            ), True
    return [], False


def _collect_parsed_models(gnn_files: list[Path]) -> list[dict[str, Any]]:
    """Parse each discovered GNN source file into per-model page data.

    Uses the reference markdown parser (``gnn.parsers.markdown_parser``);
    variables/edges keep the parsed shapes the model pages tabulate. Files
    that cannot be parsed are skipped (debug-logged) — a model page exists
    only for a successfully parsed model. The returned order matches the
    caller's ``gnn_files`` order (sorted by filename), which fixes the
    downstream slug claim order deterministically.
    """
    if not gnn_files:
        return []
    from gnn.parsers.markdown_parser import MarkdownGNNParser

    parser = MarkdownGNNParser()
    models: list[dict[str, Any]] = []
    for source in gnn_files:
        try:
            parsed = parser.parse_file(str(source))
        except (ParseError, ValueError, OSError) as e:
            logger.debug(f"Skipped unreadable GNN file {source.name}: {e}")
            continue
        if not parsed.success:
            logger.debug(f"Skipped unparseable GNN file {source.name}: {parsed.errors}")
            continue
        parsed_model = parsed.model
        from gnn.pipeline.run_context import model_provenance

        provenance = model_provenance(source, 20)
        models.append(
            {
                **provenance,
                "name": str(parsed_model.model_name),
                "source": source,
                "source_name": source.name,
                "annotation": str(parsed_model.annotation or "").strip(),
                "variables": [
                    {
                        "name": var.name,
                        "type": getattr(var.var_type, "value", str(var.var_type)),
                        "dimensions": list(var.dimensions),
                        "data_type": getattr(
                            var.data_type, "value", str(var.data_type)
                        ),
                        "description": var.description or "",
                    }
                    for var in parsed_model.variables
                ],
                "edges": [
                    {
                        "sources": list(conn.source_variables),
                        "targets": list(conn.target_variables),
                        "type": getattr(
                            conn.connection_type,
                            "value",
                            str(conn.connection_type),
                        ),
                        "annotation": conn.annotation or conn.description or "",
                    }
                    for conn in parsed_model.connections
                ],
            }
        )
    return models
