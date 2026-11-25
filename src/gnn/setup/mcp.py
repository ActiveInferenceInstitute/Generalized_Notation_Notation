"""
MCP (Model Context Protocol) integration for UV-based setup utilities.

This module exposes utility functions from the setup module through MCP,
with support for UV-based environment management and modern Python packaging.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from gnn.utils.runtime_safety.workspace_paths import (
    PathValidationError,
    resolve_repo_path,
)

logger = logging.getLogger(__name__)

# Client-supplied dependency specifications are validated before being passed
# to ``uv``. Only a PEP 508 distribution name, optional extras, and an optional
# version specifier are accepted: no shell metacharacters, no whitespace, no
# leading dashes (flag injection), no URLs or local paths.
_UV_PACKAGE_SPEC_PATTERN = re.compile(
    r"""^[A-Za-z0-9][A-Za-z0-9._-]*            # PEP 508 distribution name
        (?:\[[A-Za-z0-9][A-Za-z0-9,._-]*\])?   # optional extras list
        (?:(?:==|!=|<=|>=|~=|<|>)[A-Za-z0-9_.!+*]+)?  # optional version specifier
        $""",
    re.VERBOSE,
)
_UV_EXTRAS_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9,._-]*$")


class PackageSpecValidationError(ValueError):
    """Raised when a client-supplied package specification is rejected."""


def _validate_uv_package_spec(package_name: str) -> str:
    """Validate and normalize a client-supplied uv package specification.

    Accepts a PEP 508 distribution name with optional extras and an optional
    version specifier. Anything containing shell metacharacters, whitespace,
    leading dashes, or URL/path syntax is rejected.
    """
    if not isinstance(package_name, str) or not package_name.strip():
        raise PackageSpecValidationError("package name must be a non-empty string")
    spec = package_name.strip()
    if not _UV_PACKAGE_SPEC_PATTERN.match(spec):
        raise PackageSpecValidationError(
            f"rejected package specification {package_name!r}: must be a "
            "PEP 508 package name with optional [extras] and an optional "
            "version specifier (==, !=, <=, >=, ~=, <, >); no shell "
            "metacharacters, whitespace, or flags allowed"
        )
    return spec


def _validate_uv_extras(extras: Optional[str]) -> Optional[str]:
    """Validate a client-supplied ``uv --extras`` value."""
    if extras is None:
        return None
    if not isinstance(extras, str) or not _UV_EXTRAS_PATTERN.match(extras.strip()):
        raise PackageSpecValidationError(
            f"rejected extras {extras!r}: must be a comma-separated list of "
            "alphanumeric extra names"
        )
    return extras.strip()


logger = logging.getLogger(__name__)

# Import utilities from the setup module
from .dependency_setup import create_project_structure
from .utils import (
    ensure_directory,
    find_gnn_files,
    get_output_paths,
)
from .uv_management import get_uv_setup_info, validate_uv_setup

# MCP Tools for UV-based Setup Utilities Module


def ensure_directory_exists_mcp(directory_path: str) -> Dict[str, Any]:
    """
    Ensure a directory exists, creating it if necessary. Exposed via MCP.

    Args:
        directory_path: Directory path to ensure existence of.

    Returns:
        Dictionary with operation status and path.
    """
    try:
        path_obj = ensure_directory(Path(directory_path))
        return {
            "success": True,
            "path": str(path_obj),
            "created": not Path(
                directory_path
            ).exists(),  # Check if it was created now or existed before
        }
    except Exception as e:
        logger.error(
            f"Error in ensure_directory_exists_mcp for {directory_path}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


def find_project_gnn_files_mcp(
    search_directory: str, recursive: bool = False
) -> Dict[str, Any]:
    """
    Find all GNN (.md) files in a directory. Exposed via MCP.

    Args:
        search_directory: Directory to search.
        recursive: Whether to search recursively (default: False).

    Returns:
        Dictionary with list of found file paths or an error.
    """
    try:
        files = find_gnn_files(Path(search_directory), recursive)
        return {"success": True, "files": [str(f) for f in files], "count": len(files)}
    except Exception as e:
        logger.error(
            f"Error in find_project_gnn_files_mcp for {search_directory}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


def get_standard_output_paths_mcp(base_output_directory: str) -> Dict[str, Any]:
    """
    Get standard output paths for the pipeline. Exposed via MCP.

    Args:
        base_output_directory: Base output directory.

    Returns:
        Dictionary of named output paths or an error.
    """
    try:
        paths = get_output_paths(Path(base_output_directory))
        return {"success": True, "paths": {name: str(p) for name, p in paths.items()}}
    except Exception as e:
        logger.error(
            f"Error in get_standard_output_paths_mcp for {base_output_directory}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


def check_uv_project_status_mcp(project_directory: str) -> Dict[str, Any]:
    """
    Check the status of a UV project. Exposed via MCP.

    Args:
        project_directory: Path to the project directory.

    Returns:
        Dictionary with UV project status information.
    """
    try:
        project_root = Path(project_directory)
        status = validate_uv_setup(project_root=project_root)
        return {"success": True, "status": status}
    except Exception as e:
        logger.error(
            f"Error in check_uv_project_status_mcp for {project_directory}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


def get_uv_environment_info_mcp() -> Dict[str, Any]:
    """
    Get information about the current UV environment. Exposed via MCP.

    Returns:
        Dictionary with UV environment information.
    """
    try:
        env_info = get_uv_setup_info()
        return {"success": True, "environment_info": env_info}
    except Exception as e:
        logger.error(f"Error in get_uv_environment_info_mcp: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def setup_uv_project_structure_mcp(project_directory: str) -> Dict[str, Any]:
    """
    Set up a new UV project structure. Exposed via MCP.

    Args:
        project_directory: Path to the project directory.

    Returns:
        Dictionary with setup status.
    """
    try:
        project_root = Path(project_directory)
        success = create_project_structure(project_root, logger)
        return {
            "success": success,
            "project_directory": str(project_root),
            "message": "UV project structure setup completed"
            if success
            else "UV project structure setup failed",
        }
    except Exception as e:
        logger.error(
            f"Error in setup_uv_project_structure_mcp for {project_directory}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


def install_uv_dependency_mcp(
    package_name: str, extras: Optional[str] = None
) -> Dict[str, Any]:
    """
    Install a dependency using UV. Exposed via MCP.

    Args:
        package_name: Name of the package to install. Validated against a
            strict PEP 508 name/specifier pattern before any subprocess runs.
        extras: Optional extras to install (e.g., "dev", "ml-ai").

    Returns:
        Dictionary with installation status.
    """
    try:
        import subprocess  # nosec B404

        validated_package = _validate_uv_package_spec(package_name)
        validated_extras = _validate_uv_extras(extras)

        cmd: list[Any] = ["uv", "add", validated_package]
        if validated_extras:
            cmd.extend(["--extras", validated_extras])

        result = subprocess.run(  # nosec B603
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            return {
                "success": True,
                "package": validated_package,
                "extras": extras,
                "message": f"Successfully installed {package_name}",
            }
        else:
            return {
                "success": False,
                "package": validated_package,
                "extras": extras,
                "error": result.stderr,
                "message": f"Failed to install {validated_package}",
            }

    except PackageSpecValidationError as e:
        logger.warning(f"install_uv_dependency_mcp rejected input: {e}")
        return {
            "success": False,
            "package": package_name,
            "extras": extras,
            "error": str(e),
            "message": f"Invalid package specification: {package_name}",
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "package": package_name,
            "extras": extras,
            "error": "Installation timed out",
            "message": f"Installation of {package_name} timed out",
        }
    except Exception as e:
        logger.error(
            f"Error in install_uv_dependency_mcp for {package_name}: {e}", exc_info=True
        )
        return {
            "success": False,
            "package": package_name,
            "extras": extras,
            "error": str(e),
            "message": f"Error installing {package_name}",
        }


def sync_uv_dependencies_mcp(project_directory: str) -> Dict[str, Any]:
    """
    Sync dependencies using UV. Exposed via MCP.

    Args:
        project_directory: Path to the project directory. Resolved through
            repo containment checks before use as the subprocess cwd.

    Returns:
        Dictionary with sync status.
    """
    try:
        import subprocess  # nosec B404

        project_root = resolve_repo_path(
            project_directory,
            purpose="Project directory",
            must_exist=True,
        )
        result = subprocess.run(  # nosec B607 B603
            ["uv", "sync"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes timeout
        )

        if result.returncode == 0:
            return {
                "success": True,
                "project_directory": str(project_root),
                "message": "Dependencies synced successfully",
            }
        else:
            return {
                "success": False,
                "project_directory": str(project_root),
                "error": result.stderr,
                "message": "Failed to sync dependencies",
            }

    except PathValidationError as e:
        logger.warning(f"sync_uv_dependencies_mcp rejected path: {e}")
        return {
            "success": False,
            "project_directory": str(project_directory),
            "error": str(e),
            "message": f"Invalid project directory: {project_directory}",
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "project_directory": str(project_directory),
            "error": "Sync timed out",
            "message": "Dependency sync timed out",
        }
    except Exception as e:
        logger.error(
            f"Error in sync_uv_dependencies_mcp for {project_directory}: {e}",
            exc_info=True,
        )
        return {
            "success": False,
            "project_directory": str(project_directory),
            "error": str(e),
            "message": "Error syncing dependencies",
        }


# MCP Registration Function
def register_tools(mcp_instance: Any) -> None:
    """Register UV-based setup utility tools with the MCP."""

    # Generic namespaced tools
    mcp_instance.register_tool(
        "ensure_directory_exists",
        ensure_directory_exists_mcp,
        {
            "directory_path": {
                "type": "string",
                "description": "Path of the directory to create if it doesn't exist.",
            }
        },
        "Ensures a directory exists, creating it if necessary. Returns the absolute path.",
    )

    mcp_instance.register_tool(
        "find_project_gnn_files",
        find_project_gnn_files_mcp,
        {
            "search_directory": {
                "type": "string",
                "description": "The directory to search for GNN (.md) files.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Set to true to search recursively. Defaults to false.",
                "optional": True,
            },
        },
        "Finds all GNN (.md) files in a specified directory within the project.",
    )

    mcp_instance.register_tool(
        "get_standard_output_paths",
        get_standard_output_paths_mcp,
        {
            "base_output_directory": {
                "type": "string",
                "description": "The base directory where output subdirectories will be managed.",
            }
        },
        "Gets a dictionary of standard output directory paths (e.g., for type_check, visualization), creating them if needed.",
    )

    mcp_instance.register_tool(
        "check_uv_project_status",
        check_uv_project_status_mcp,
        {
            "project_directory": {
                "type": "string",
                "description": "Path to the project directory to check.",
            }
        },
        "Checks the status of a UV project including pyproject.toml, uv.lock, and virtual environment.",
    )

    mcp_instance.register_tool(
        "get_uv_environment_info",
        get_uv_environment_info_mcp,
        {},
        "Gets information about the current UV environment including paths and status.",
    )

    mcp_instance.register_tool(
        "setup_uv_project_structure",
        setup_uv_project_structure_mcp,
        {
            "project_directory": {
                "type": "string",
                "description": "Path to the project directory to set up.",
            }
        },
        "Sets up a new UV project structure with standard directories and configuration.",
    )

    mcp_instance.register_tool(
        "install_uv_dependency",
        install_uv_dependency_mcp,
        {
            "package_name": {
                "type": "string",
                "description": "Name of the package to install.",
            },
            "extras": {
                "type": "string",
                "description": "Optional extras to install (e.g., 'dev', 'ml-ai').",
                "optional": True,
            },
        },
        "Installs a dependency using UV with optional extras support.",
    )

    mcp_instance.register_tool(
        "sync_uv_dependencies",
        sync_uv_dependencies_mcp,
        {
            "project_directory": {
                "type": "string",
                "description": "Path to the project directory to sync dependencies for.",
            }
        },
        "Syncs dependencies using UV from pyproject.toml and updates the lock file.",
    )

    logger.info("UV-based setup module MCP tools registered.")
