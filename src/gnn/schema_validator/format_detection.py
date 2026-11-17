"""Format detection and binary (pickle) parsing helpers for GNNParser.

Holds FormatDetectionMixin: extension/content-based format detection plus
binary-file ingestion, split out of gnn/schema_validator/syntax.py.
"""

import logging
import re
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from gnn.types import ParsedGNN

logger = logging.getLogger(__name__)


class FormatDetectionMixin:
    """Mixin with GNN file-format detection and binary parsing methods."""

    FORMAT_SIGNATURES: MappingProxyType[str, tuple[Any, ...]]

    def _detect_file_format(self, file_path: Path) -> str:
        """Enhanced format detection with content analysis."""
        # First try extension-based detection
        extension = file_path.suffix.lower()
        extension_map: dict[str, Any] = {
            ".md": "markdown",
            ".json": "json",
            ".xml": "xml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".pkl": "pkl",
            ".pickle": "binary",
        }

        if extension in extension_map:
            detected = extension_map[extension]

            # Verify with content analysis for ambiguous cases
            if extension in [".md", ".txt"] and file_path.exists():
                content_format = self._detect_format_from_content(file_path)
                if content_format != "markdown":
                    return content_format

            return cast("str", detected)

        # Content-based detection for unknown extensions
        return self._detect_format_from_content(file_path)

    def _detect_format_from_content(self, file_path: Path) -> str:
        """Detect format from file content analysis."""
        try:
            # Try reading as text first
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read(2000)  # Read first 2KB

            content.lower()

            # Check format signatures
            for fmt, patterns in self.FORMAT_SIGNATURES.items():
                if fmt == "binary":
                    continue  # Skip binary patterns for text content

                text_patterns = [
                    pattern for pattern in patterns if isinstance(pattern, str)
                ]
                if any(
                    re.search(pattern, content, re.IGNORECASE)
                    for pattern in text_patterns
                ):
                    return fmt

            # Check for GNN markdown indicators
            if "##" in content and any(
                section in content
                for section in ["GNNSection", "ModelName", "StateSpaceBlock"]
            ):
                return "markdown"

            return "markdown"  # Default recovery

        except UnicodeDecodeError:
            return "binary"
        except Exception as e:
            logger.debug(f"Format detection failed for file: {e}")
            return "unknown"

    def _parse_binary_file(self, file_path: Path) -> ParsedGNN:
        """Parse binary files (pickle format)."""
        try:
            from gnn.parsers.binary_parser import safe_pickle_load

            with open(file_path, "rb") as f:
                data = safe_pickle_load(f)

            # Convert pickle data to ParsedGNN format
            return self._convert_pickle_to_parsed_gnn(data)

        except Exception as e:
            # Create minimal parsed representation for failed binary files
            return ParsedGNN(
                gnn_section="BinaryGNN",
                version="1.0",
                model_name=f"BinaryModel_{file_path.stem}",
                model_annotation=f"Binary file: {file_path.name}",
                variables={},
                connections=[],
                parameters={},
                equations=[],
                time_config={},
                ontology_mappings={},
                model_parameters={},
                footer="",
                metadata={"parse_error": str(e), "source_format": "binary"},
            )

    def _convert_pickle_to_parsed_gnn(self, data: Any) -> ParsedGNN:
        """Convert pickle data to ParsedGNN structure."""
        # Implementation depends on pickle data structure
        # This is a simplified version
        if isinstance(data, dict):
            return ParsedGNN(
                gnn_section=data.get("gnn_section", "PickleGNN"),
                version=data.get("version", "1.0"),
                model_name=data.get("model_name", "PickleModel"),
                model_annotation=data.get("annotation", ""),
                variables=data.get("variables", {}),
                connections=data.get("connections", []),
                parameters=data.get("parameters", {}),
                equations=data.get("equations", []),
                time_config=data.get("time_config", {}),
                ontology_mappings=data.get("ontology_mappings", {}),
                model_parameters=data.get("model_parameters", {}),
                footer=data.get("footer", ""),
                source_format="pickle",
            )
        else:
            # Create minimal representation for non-dict pickle data
            return ParsedGNN(
                gnn_section="PickleGNN",
                version="1.0",
                model_name="PickleModel",
                model_annotation=f"Pickled data: {type(data).__name__}",
                variables={},
                connections=[],
                parameters={},
                equations=[],
                time_config={},
                ontology_mappings={},
                model_parameters={},
                footer="",
                source_format="pickle",
            )
