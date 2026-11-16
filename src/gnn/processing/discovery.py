#!/usr/bin/env python3
"""
GNN File Discovery Strategy Module

This module provides intelligent file discovery with content analysis
and validation for GNN models.
"""

import hashlib
import logging
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NON_MODEL_MARKDOWN_FILENAMES = {
    "agents.md",
    "changelog.md",
    "contributing.md",
    "index.md",
    "license.md",
    "readme.md",
}
NON_MODEL_MARKDOWN_SUFFIXES = (
    ".example.md",
    ".template.md",
)


def is_model_source_path(file_path: Path) -> bool:
    """Return True when a path is a maintained GNN model source."""
    name = file_path.name.lower()
    if name in NON_MODEL_MARKDOWN_FILENAMES:
        return False
    return not any(name.endswith(suffix) for suffix in NON_MODEL_MARKDOWN_SUFFIXES)


def model_artifact_stems(root: Path, sources: Iterable[Path]) -> dict[Path, str]:
    """Keep unique stems and distinguish colliding source paths on any platform."""

    def portable_key(stem: str) -> str:
        return unicodedata.normalize("NFC", stem).casefold()

    paths = sorted(sources)
    counts = Counter(portable_key(path.stem) for path in paths)
    stems = {
        path: path.stem
        for path in paths
        if counts[portable_key(path.stem)] == 1 and path.stem not in {".", ".."}
    }
    reserved = {portable_key(stem) for stem in stems.values()}
    for path in paths:
        if path in stems:
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(relative.encode()).hexdigest()[:12]
        label = re.sub(r"[^A-Za-z0-9_-]", "_", path.stem)[:48]
        stem = f"{label}-{digest}"
        suffix = 1
        while portable_key(stem) in reserved:
            stem = f"{label}-{digest}-{suffix}"
            suffix += 1
        stems[path] = stem
        reserved.add(portable_key(stem))
    return stems


def recorded_artifact_stem(value: Any) -> Optional[str]:
    """Accept the portable override stored for colliding standalone sources."""
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    return None


@dataclass
class DiscoveryResult:
    """Result of file discovery operation."""

    files_found: List[Path]
    analysis_time: float
    file_types: Dict[str, int]
    potential_gnn_files: List[Path]
    metadata: Dict[str, Any]


class FileDiscoveryStrategy:
    """
    Intelligent file discovery strategy for GNN models.

    Performs content-aware discovery and basic analysis
    to identify potential GNN files.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.target_extensions = [".md", ".json", ".xml", ".yaml", ".pkl"]
        self.gnn_indicators = [
            "## ModelName",
            "## StateSpaceBlock",
            "## Connections",
            "GNN",
            "model_name",
            "variables",
            "connections",
        ]

    def configure(
        self, target_extensions: Optional[List[str]] = None, **kwargs: Any
    ) -> Any:
        """Configure discovery parameters."""
        if target_extensions:
            self.target_extensions = target_extensions

    def discover(self, target_dir: Path) -> List[Path]:
        """
        Discover GNN files in target directory.

        Args:
            target_dir: Directory to search

        Returns:
            List of discovered file paths
        """
        logger.info(f"Starting file discovery in {target_dir}")
        start_time = time.time()

        # Basic file discovery
        all_files: list[Any] = []
        potential_gnn_files: list[Any] = []

        # Recursive search for target extensions
        for ext in self.target_extensions:
            pattern = f"**/*{ext}"
            found_files = list(target_dir.rglob(pattern))
            all_files.extend(found_files)
            logger.debug(f"Found {len(found_files)} {ext} files")

        # Content analysis for GNN indicators
        for file_path in all_files:
            if is_model_source_path(file_path) and self._analyze_file_content(
                file_path
            ):
                potential_gnn_files.append(file_path)

        discovery_time = time.time() - start_time
        logger.info(
            f"Discovery completed: {len(potential_gnn_files)}/{len(all_files)} potential GNN files in {discovery_time:.3f}s"
        )

        return potential_gnn_files

    def _analyze_file_content(self, file_path: Path) -> bool:
        """
        Analyze file content for GNN indicators.

        Args:
            file_path: Path to analyze

        Returns:
            bool: Whether file appears to be a GNN file
        """
        try:
            # Read file content
            content = file_path.read_text(encoding="utf-8", errors="ignore")

            # Check for GNN indicators
            indicators_found = 0
            for indicator in self.gnn_indicators:
                if indicator.lower() in content.lower():
                    indicators_found += 1

            # Require at least 2 indicators for GNN classification
            return indicators_found >= 2

        except Exception as e:
            logger.debug(f"Could not analyze {file_path}: {e}")
            return False
