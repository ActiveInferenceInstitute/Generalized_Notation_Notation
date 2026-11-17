"""
Utils Module MCP Integration

This module exposes utility functions and system information tools through the Model Context Protocol.
It provides tools for system diagnostics, file operations, logging management, and general utilities.

Key Features:
- System information and diagnostics
- File and directory operations
- Logging management and configuration
- Performance tracking and monitoring
- Environment and dependency validation
- Utility function access
"""

import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict

import psutil

# Substrings whose presence in an environment-variable name (lowercased)
# marks the variable as sensitive for reporting purposes. Centralized so the
# same policy applies anywhere utils reports environment contents.
SENSITIVE_ENV_KEY_MARKERS: tuple[str, ...] = (
    "password",
    "passwd",
    "secret",
    "key",
    "token",
    "credential",
    "auth",
)


def is_sensitive_env_key(key: str) -> bool:
    """Return True when *key* looks like a secret-carrying environment name.

    Case-insensitive substring match against
    :data:`SENSITIVE_ENV_KEY_MARKERS`; ``AWS_SESSION_TOKEN``,
    ``GOOGLE_API_KEY``, ``DATABASE_CREDENTIALS`` etc. are all caught.
    """
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_ENV_KEY_MARKERS)


def redact_environment() -> dict[str, str]:
    """Return a copy of ``os.environ`` with sensitive variables removed.

    The returned mapping is safe to embed in reports or diagnostics;
    variables whose name contains any sensitive marker are omitted entirely
    (never value-masked) so report consumers cannot recover partial secrets.
    """
    return {
        key: value for key, value in os.environ.items() if not is_sensitive_env_key(key)
    }


logger = logging.getLogger(__name__)


def get_system_info(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Get comprehensive system information and diagnostics.

    Returns:
        Dictionary containing system information.
    """
    try:
        # System information
        system_info: dict[str, Any] = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "python_path": sys.path,
        }

        # Memory information
        memory = psutil.virtual_memory()
        memory_info: dict[str, Any] = {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
        }

        # CPU frequency is optional on several kernels (including Apple
        # Silicon). A failed probe must not discard the other diagnostics.
        frequency = None
        frequency_error = None
        try:
            sampled_frequency = psutil.cpu_freq()
            if sampled_frequency is not None:
                frequency = sampled_frequency._asdict()
        except (OSError, RuntimeError, SystemError, psutil.Error) as error:
            frequency_error = str(error)

        # CPU information
        cpu_info: dict[str, Any] = {
            "count": psutil.cpu_count(),
            "count_logical": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=1),
            "frequency": frequency,
        }

        if frequency_error is not None:
            cpu_info["frequency_error"] = frequency_error

        # Disk information
        disk = psutil.disk_usage("/")
        disk_info: dict[str, Any] = {
            "total": disk.total,
            "used": disk.used,
            "free": disk.free,
            "percent": disk.percent,
            "total_gb": round(disk.total / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
        }

        return {
            "success": True,
            "system": system_info,
            "memory": memory_info,
            "cpu": cpu_info,
            "disk": disk_info,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"success": False, "error": str(e)}


def get_environment_info(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Get environment information including Python environment and dependencies.

    Returns:
        Dictionary containing environment information.
    """
    try:
        # Environment variables
        env_vars: dict[str, str] = {}
        for key, value in os.environ.items():
            if is_sensitive_env_key(key):
                continue
            env_vars[key] = value

        # Python packages
        # importlib.metadata enumerates installed distributions (pkg_resources is retired).
        from importlib import metadata

        installed_packages: list[Any] = []
        for dist in metadata.distributions():
            installed_packages.append(
                {
                    "name": dist.metadata["Name"],
                    "version": dist.version,
                    "location": str(dist.locate_file("")),
                }
            )

        # Working directory and paths
        current_dir = Path.cwd()
        home_dir = Path.home()

        return {
            "success": True,
            "environment_variables": env_vars,
            "installed_packages": installed_packages,
            "working_directory": str(current_dir),
            "home_directory": str(home_dir),
            "python_path": sys.path,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.error(f"Error getting environment info: {e}")
        return {"success": False, "error": str(e)}


def get_file_info(mcp_instance_ref: Any, file_path: str) -> Dict[str, Any]:
    """
    Get detailed information about a file or directory.

    Args:
        file_path: Path to the file or directory.

    Returns:
        Dictionary containing file information.
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {file_path}"}

        stat = path.stat()

        file_info: dict[str, Any] = {
            "path": str(path),
            "name": path.name,
            "parent": str(path.parent),
            "exists": True,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "is_symlink": path.is_symlink(),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024**2), 2),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "permissions": oct(stat.st_mode)[-3:],
            "owner": stat.st_uid,
            "group": stat.st_gid,
        }

        # Additional info for directories
        if path.is_dir():
            try:
                contents = list(path.iterdir())
                file_info["contents_count"] = len(contents)
                file_info["contents"] = [
                    item.name for item in contents[:10]
                ]  # First 10 items
                if len(contents) > 10:
                    file_info["contents"].append(f"... and {len(contents) - 10} more")
            except PermissionError:
                file_info["contents_error"] = "Permission denied"

        return {"success": True, "file_info": file_info}
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {"success": False, "error": str(e)}


def get_logging_info(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Get current logging configuration and status.

    Returns:
        Dictionary containing logging information.
    """
    try:
        # Get root logger info
        root_logger = logging.getLogger()

        # Get all loggers
        loggers: dict[Any, Any] = {}
        for name in logging.root.manager.loggerDict:
            logger_obj = logging.getLogger(name)
            loggers[name] = {
                "level": logging.getLevelName(logger_obj.level),
                "handlers": len(logger_obj.handlers),
                "propagate": logger_obj.propagate,
            }

        # Get current logging configuration
        config: dict[str, Any] = {
            "root_level": logging.getLevelName(root_logger.level),
            "root_handlers": len(root_logger.handlers),
            "loggers_count": len(loggers),
            "loggers": loggers,
        }

        return {"success": True, "logging_config": config}
    except Exception as e:
        logger.error(f"Error getting logging info: {e}")
        return {"success": False, "error": str(e)}


def validate_dependencies(mcp_instance_ref: Any) -> Dict[str, Any]:
    """
    Validate system dependencies and requirements.

    Returns:
        Dictionary containing dependency validation results.
    """
    try:
        from gnn.utils.runtime_safety.dependency_validator import (
            validate_pipeline_dependencies,
        )

        # Validate pipeline dependencies
        dependency_validation = validate_pipeline_dependencies()

        # Check for common Python packages
        required_packages: list[Any] = [
            "numpy",
            "matplotlib",
            "networkx",
            "pandas",
            "requests",
            "pathlib",
            "json",
            "logging",
            "typing",
            "psutil",
        ]

        package_status: dict[Any, Any] = {}
        for package in required_packages:
            try:
                __import__(package)
                package_status[package] = "available"
            except ImportError:
                package_status[package] = "missing"

        return {
            "success": True,
            "dependency_validation": dependency_validation,
            "package_status": package_status,
            "missing_packages": [
                pkg for pkg, status in package_status.items() if status == "missing"
            ],
        }
    except Exception as e:
        logger.error(f"Error validating dependencies: {e}")
        return {"success": False, "error": str(e)}


def register_tools(mcp_instance: Any) -> Any:
    """
    Register utility tools with the MCP server.

    Args:
        mcp_instance: The MCP instance to register tools with.
    """
    logger.info("Registering utils MCP tools")

    # Named wrappers keep MCP audit checks happy (no functools.partial names).
    def get_system_info_tool() -> Any:
        """Return system info tool."""
        return get_system_info(mcp_instance)

    def get_environment_info_tool() -> Any:
        """Return environment info tool."""
        return get_environment_info(mcp_instance)

    def get_file_info_tool(file_path: str) -> Any:
        """Return file info tool."""
        return get_file_info(mcp_instance, file_path)

    def get_logging_info_tool() -> Any:
        """Return logging info tool."""
        return get_logging_info(mcp_instance)

    def validate_dependencies_tool() -> Any:
        """Validate dependencies tool."""
        return validate_dependencies(mcp_instance)

    # Register tools
    mcp_instance.register_tool(
        name="get_system_info",
        function=get_system_info_tool,
        schema={},
        description="Get comprehensive system information including CPU, memory, disk, and platform details.",
    )

    mcp_instance.register_tool(
        name="get_environment_info",
        function=get_environment_info_tool,
        schema={},
        description="Get environment information including Python packages, environment variables, and paths.",
    )

    mcp_instance.register_tool(
        name="get_file_info",
        function=get_file_info_tool,
        schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file or directory to analyze",
                }
            },
            "required": ["file_path"],
        },
        description="Get detailed information about a file or directory including size, permissions, and contents.",
    )

    mcp_instance.register_tool(
        name="get_logging_info",
        function=get_logging_info_tool,
        schema={},
        description="Get current logging configuration and status for all loggers.",
    )

    mcp_instance.register_tool(
        name="validate_dependencies",
        function=validate_dependencies_tool,
        schema={},
        description="Validate system dependencies and required packages for the GNN pipeline.",
    )

    logger.info("Successfully registered utils MCP tools")
