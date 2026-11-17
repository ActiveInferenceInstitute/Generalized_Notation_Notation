#!/usr/bin/env python3
"""
Simple GNN Validator Module

This module provides a simplified validator for GNN files without relying on
complex dependencies or circular imports. It's designed to be used as a recovery
when the full validation system encounters issues.
"""

import logging
from pathlib import Path
from typing import Any, Dict, cast

logger = logging.getLogger(__name__)


class SimpleValidator:
    """
    Simple validator for GNN files.

    This validator performs basic checks without complex dependencies.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.valid_extensions = [".md", ".json", ".xml", ".yaml", ".pkl", ".pickle"]

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a GNN file with basic checks.

        Args:
            file_path: Path to the file to validate

        Returns:
            Dictionary with validation results
        """
        result: dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "format": self._detect_format(file_path),
        }

        # Check if file exists
        if not file_path.exists():
            result["is_valid"] = False
            result["errors"].append(f"File does not exist: {file_path}")
            return result

        # Check if file is readable
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(f"Could not read file: {e}")
            return result

        # Check for basic GNN markers
        gnn_markers: list[Any] = [
            "model",
            "gnn",
            "variable",
            "connection",
            "ModelName",
            "StateSpaceBlock",
        ]
        found_markers = [marker for marker in gnn_markers if marker in content]

        if not found_markers:
            result["warnings"].append("No GNN markers found in file")

        # Check for section structure in markdown files
        if file_path.suffix.lower() == ".md":
            sections: list[Any] = ["ModelName", "StateSpaceBlock", "Connections"]
            missing_sections = [
                section for section in sections if section not in content
            ]

            if missing_sections:
                result["warnings"].append(
                    f"Missing sections: {', '.join(missing_sections)}"
                )

        return result

    def _detect_format(self, file_path: Path) -> str:
        """Detect file format from extension."""
        ext = file_path.suffix.lower()

        format_map: dict[str, Any] = {
            ".md": "markdown",
            ".json": "json",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".pkl": "pkl",
            ".pickle": "pickle",
        }

        return cast("str", format_map.get(ext, "unknown"))

    def validate_directory(
        self, directory: Path, recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Validate all GNN files in a directory.

        Args:
            directory: Directory to validate
            recursive: Whether to search recursively

        Returns:
            Dictionary with validation results
        """
        results: dict[str, Any] = {
            "directory": str(directory),
            "files_validated": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "file_results": {},
        }

        # Find files to validate
        if recursive:
            files: list[Any] = []
            for ext in self.valid_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            files = []
            for ext in self.valid_extensions:
                files.extend(directory.glob(f"*{ext}"))

        # Validate each file
        for file_path in files:
            file_result = self.validate_file(file_path)
            results["files_validated"] += 1

            if file_result["is_valid"]:
                results["valid_files"] += 1
            else:
                results["invalid_files"] += 1

            results["file_results"][str(file_path)] = file_result

        return results


def check_gnn_file_basic(file_path: Path) -> Dict[str, Any]:
    """
    Convenience function to validate a GNN file.

    Args:
        file_path: Path to the file to validate

    Returns:
        Dictionary with validation results
    """
    validator = SimpleValidator()
    return validator.validate_file(file_path)


def check_gnn_directory_basic(
    directory: Path, recursive: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to validate all GNN files in a directory.

    Args:
        directory: Directory to validate
        recursive: Whether to search recursively

    Returns:
        Dictionary with validation results
    """
    validator = SimpleValidator()
    return validator.validate_directory(directory, recursive)
