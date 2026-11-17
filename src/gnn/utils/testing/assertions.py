"""Assertion helpers shared by the test suite.

Moved verbatim from ``gnn/utils/testing_utils.py`` (S2-33 Step 1)."""

import json
from pathlib import Path
from typing import Any, Dict


def assert_file_exists(file_path: Path, message: str = "") -> None:
    """Assert that a file exists."""
    if not file_path.exists():
        raise AssertionError(f"File does not exist: {file_path}. {message}")


def assert_valid_json(file_path: Path) -> None:
    """Assert that a file contains valid JSON."""
    try:
        with open(file_path) as f:
            json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise AssertionError(
            f"File does not contain valid JSON: {file_path}. Error: {e}"
        ) from e


def assert_directory_structure(
    base_dir: Path, expected_structure: Dict[str, Any]
) -> None:
    """Assert that a directory has the expected structure."""
    for item_name, item_content in expected_structure.items():
        item_path = base_dir / item_name

        if isinstance(item_content, dict):
            # This is a directory
            if not item_path.exists():
                raise AssertionError(f"Directory does not exist: {item_path}")
            if not item_path.is_dir():
                raise AssertionError(f"Path is not a directory: {item_path}")

            # Recursively check subdirectories
            assert_directory_structure(item_path, item_content)
        else:
            # This is a file
            if not item_path.exists():
                raise AssertionError(f"File does not exist: {item_path}")
