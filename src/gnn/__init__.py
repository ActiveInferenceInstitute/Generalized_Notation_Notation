"""
GNN module for GNN Processing Pipeline.

This module provides GNN file discovery, parsing, and validation capabilities.

Importing ``gnn`` is intentionally LIGHT (lazy PEP 562 re-exports): no
submodule executes at import time, so the package can be imported — and the
headless POMDP extractor (``gnn.extract.pomdp_extractor``) used — without pulling in
the full pipeline stack or heavy module-scope dependencies (psutil,
matplotlib). Names resolve through ``__getattr__`` on first access.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Static re-export surface for type checkers: mirrors the pre-lazy eager
    # imports so mypy resolves names without executing submodules at runtime.
    from .execute.executor import GNNExecutor, execute_gnn_model
    from .export.processor import generate_exports
    from .model_registry.registry import ModelRegistry
    from .parsers.basic import (
        GNNFormalParser,
        ParsedGNNFormal,
        get_parse_tree_visualization,
        parse_gnn_formal,
        validate_gnn_syntax,
    )
    from .parsers.common import GNNFormat
    from .parsers.system import GNNParsingSystem
    from .pipeline import run_pipeline
    from .processing.multi_format_processor import process_gnn_multi_format
    from .processing.processor import (
        check_gnn_file_structure,
        discover_gnn_files,
        generate_gnn_report,
        get_module_info,
        parse_gnn_file,
        process_gnn_directory,
        process_gnn_directory_lightweight,
    )
    from .render.processor import get_available_renderers, render_gnn_spec
    from .types import ParsedGNN, ValidationLevel
    from .utils.config_io.config_loader import GNNPipelineConfig, load_config

__version__ = "3.6.0"

# Ensure tests see MCP feature presence consistently
FEATURES: dict[str, Any] = {
    "file_discovery": True,
    "content_parsing": True,
    "structure_validation": True,
    "report_generation": True,
    "core_validation": True,
    "mcp_integration": True,
    "pipeline_orchestration": True,
}

# Explicit name -> source submodule map for every re-export. Resolving a name
# imports only that one submodule.
_EXPORT_MAP: dict[str, str] = {
    # multi_format_processor
    "process_gnn_multi_format": "processing.multi_format_processor",
    # parsers.basic
    "GNNFormalParser": "parsers.basic",
    "ParsedGNNFormal": "parsers.basic",
    "get_parse_tree_visualization": "parsers.basic",
    "parse_gnn_formal": "parsers.basic",
    "validate_gnn_syntax": "parsers.basic",
    # parsers.common
    "GNNFormat": "parsers.common",
    # parsers.system — canonical 23-format registry (23 parsers, 22 serializers;
    # PNML is parse-only)
    "GNNParsingSystem": "parsers.system",
    # processor
    "discover_gnn_files": "processing.processor",
    "generate_gnn_report": "processing.processor",
    "get_module_info": "processing.processor",
    "parse_gnn_file": "processing.processor",
    "process_gnn_directory": "processing.processor",
    "process_gnn_directory_lightweight": "processing.processor",
    "check_gnn_file_structure": "processing.processor",
    # types — canonical domain types
    "ParsedGNN": "types",
    "ValidationLevel": "types",
    # Cross-package programmatic entry points — the root package is the
    # one aggregate facade for the pipeline, execution, rendering, export,
    # model-registry, and configuration surfaces.
    "run_pipeline": "pipeline",
    "GNNExecutor": "execute.executor",
    "execute_gnn_model": "execute.executor",
    "render_gnn_spec": "render.processor",
    "get_available_renderers": "render.processor",
    "generate_exports": "export.processor",
    "ModelRegistry": "model_registry.registry",
    "GNNPipelineConfig": "utils.config_io.config_loader",
    "load_config": "utils.config_io.config_loader",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve a re-exported name (PEP 562).

    The submodule owning ``name`` is imported on first access and the value is
    cached in the module globals. Any ImportError raised while importing the
    owning submodule propagates unchanged — no silent fallback.
    """
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    # Cache so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include re-exported names in ``dir(gnn)`` alongside module globals."""
    return sorted(set(globals()) | set(_EXPORT_MAP))


def validate_gnn_source(source: Any, *, is_content: bool = False) -> Any:
    """Validate a GNN source (file path or content string).

    Args:
        source: File path (str or Path) or raw GNN content string.
        is_content: If True, treat source as raw content regardless of type.

    Returns:
        Dict with keys ``is_valid`` (bool) and ``errors`` (list[str]).
    """
    from pathlib import Path as _Path

    if not is_content and isinstance(source, (str, _Path)) and _Path(source).exists():
        content = _Path(source).read_text(encoding="utf-8")
    else:
        content = str(source)
    from .parsers.basic import validate_gnn_syntax as _validate_gnn_syntax

    is_valid, errors = _validate_gnn_syntax(content)
    return {"is_valid": is_valid, "errors": errors}


__all__: list[str] = [
    # Processor functions
    "process_gnn_directory_lightweight",
    "discover_gnn_files",
    "parse_gnn_file",
    "check_gnn_file_structure",
    "process_gnn_directory",
    "generate_gnn_report",
    "get_module_info",
    # Multi-format processor
    "process_gnn_multi_format",
    # Parser classes and functions
    "ValidationLevel",
    "ParsedGNN",
    "GNNParsingSystem",
    "GNNFormat",
    "GNNFormalParser",
    "ParsedGNNFormal",
    "parse_gnn_formal",
    "get_parse_tree_visualization",
    "validate_gnn_syntax",
    "__version__",
    "FEATURES",
    "validate_gnn_source",
    # Cross-package programmatic entry points
    "run_pipeline",
    "GNNExecutor",
    "execute_gnn_model",
    "render_gnn_spec",
    "get_available_renderers",
    "generate_exports",
    "ModelRegistry",
    "GNNPipelineConfig",
    "load_config",
]
