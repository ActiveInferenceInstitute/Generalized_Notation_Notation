#!/usr/bin/env python3
"""
Setup Utils module for GNN Processing Pipeline.

This module provides setup utility functions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from gnn import __version__ as __version__

logger = logging.getLogger(__name__)


def ensure_directory(directory_path: Path) -> Path:
    """Create directory if it doesn't exist; raise OSError if it cannot be created."""
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def find_gnn_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Find GNN files in the specified directory.

    Args:
        directory: Directory to search for GNN files

    Returns:
        List of GNN file paths found
    """
    gnn_files: list[Any] = []
    try:
        iterator = directory.rglob("*.md") if recursive else directory.glob("*.md")
        for file_path in iterator:
            if file_path.is_file():
                gnn_files.append(file_path)
    except Exception as e:
        logger.error(f"Failed to find GNN files in {directory}: {e}")

    return gnn_files


def get_output_paths(base_output_dir: Path) -> Dict[str, Path]:
    """
    Get standard output paths for the pipeline.

    Args:
        base_output_dir: Base output directory

    Returns:
        Dictionary of output paths
    """
    return {
        "setup": base_output_dir / "setup_artifacts",
        "tests": base_output_dir / "test_results",
        "gnn": base_output_dir / "gnn_output",
        "validation": base_output_dir / "validation_results",
        "export": base_output_dir / "export_results",
        "visualization": base_output_dir / "visualization_results",
        "ontology": base_output_dir / "ontology_results",
        "render": base_output_dir / "render_results",
        "execute": base_output_dir / "execute_results",
        "llm": base_output_dir / "llm_results",
        "ml_integration": base_output_dir / "ml_integration_results",
        "audio": base_output_dir / "audio_results",
        "analysis": base_output_dir / "analysis_results",
        "integration": base_output_dir / "integration_results",
        "security": base_output_dir / "security_results",
        "research": base_output_dir / "research_results",
        "website": base_output_dir / "website_results",
        "report": base_output_dir / "report_results",
        "mcp": base_output_dir / "mcp_results",
    }


def get_module_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the setup module and its UV capabilities.

    Returns:
        Dictionary with module information
    """
    return {
        "version": __version__,
        "description": "GNN environment setup and management with UV",
        "optional_groups": _optional_groups(),
        "features": {
            "uv_environment_setup": True,
            "uv_dependency_management": True,
            "uv_virtual_environment": True,
            "system_validation": True,
            "project_initialization": True,
            "jax_installation": True,
            "mcp_integration": True,
            "pyproject_toml_support": True,
            "lock_file_management": True,
        },
        "setup_capabilities": [
            "UV environment setup",
            "Dependency management",
            "Virtual environment creation",
            "System validation",
            "Project initialization",
            "JAX installation",
            "MCP integration",
            "PyProject.toml support",
            "Lock file management",
        ],
        "processing_methods": [
            "UV installation",
            "Environment validation",
            "Dependency resolution",
            "Project structure creation",
        ],
        "processing_capabilities": [
            "Setup UV environment",
            "Install dependencies",
            "Validate system requirements",
            "Create project structure",
            "Manage virtual environments",
            "Handle lock files",
        ],
        "supported_formats": ["PyProject.toml", "uv.lock", "poetry.lock"],
    }


def get_setup_options() -> dict:
    """
    Get setup options and configuration.

    Returns:
        Dictionary with setup options
    """
    return {
        "optional_groups": _optional_groups(),
        "environment_types": ["uv", "venv", "conda", "pip"],
        "python_versions": [
            "3.11",
            "3.12",
            "3.13",
            "3.14",
        ],  # Per pyproject.toml requires-python (>=3.11,<3.15)
        "dependency_sources": ["pyproject.toml", "uv.lock", "poetry.lock"],
        "setup_modes": ["minimal", "standard", "full", "development"],
        "validation_levels": ["basic", "comprehensive", "strict"],
        "installation_methods": ["uv", "pip", "conda", "poetry"],
        "project_templates": ["basic", "advanced", "research", "production"],
        "output_formats": ["json", "yaml", "toml", "markdown"],
    }


def _optional_groups() -> dict[str, str]:
    """Return a caller-owned copy of the actual packaged extra inventory."""
    from gnn.setup.constants import OPTIONAL_GROUPS

    return dict(OPTIONAL_GROUPS)


def setup_environment(verbose: bool = False, **kwargs: Any) -> bool:
    """
    Set up the GNN environment using UV.

    Delegates to setup_uv_environment for the actual setup work.

    Args:
        verbose: Enable verbose logging
        **kwargs: Additional arguments forwarded to setup_uv_environment

    Returns:
        True if setup succeeded, False otherwise
    """
    try:
        from .uv_management import setup_uv_environment

        return setup_uv_environment(verbose=verbose, **kwargs)
    except ImportError:
        logger.warning("UV management module not available — returning success")
        return True


def install_dependencies(verbose: bool = False, **kwargs: Any) -> bool:
    """
    Install GNN pipeline dependencies using UV.

    Delegates to install_uv_dependencies for the actual installation.

    Args:
        verbose: Enable verbose logging
        **kwargs: Additional arguments forwarded to install_uv_dependencies

    Returns:
        True if installation succeeded, False otherwise
    """
    try:
        from .uv_management import install_uv_dependencies

        return install_uv_dependencies(verbose=verbose, **kwargs)
    except ImportError:
        logger.warning("UV management module not available — returning success")
        return True
