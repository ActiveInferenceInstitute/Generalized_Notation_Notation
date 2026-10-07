"""Tests for the processing module's public API surface.

Covers: package exports, parse_gnn_file, and check_gnn_file_structure with
inline content so no files need to be written.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestProcessingPackageSurface:
    """Test the package-level public surface."""

    def test_package_exports_processing_surface(self) -> None:
        import gnn.processing as processing

        for name in processing.__all__:
            assert hasattr(processing, name)
        for name in (
            "FileDiscoveryStrategy",
            "GNNProcessor",
            "ProcessingContext",
            "ProcessingPhase",
            "check_gnn_file_structure",
            "discover_gnn_files",
            "is_model_source_path",
            "parse_gnn_file",
            "process_gnn_directory_lightweight",
        ):
            assert name in processing.__all__


class TestLightweightParsing:
    """Test parse_gnn_file and check_gnn_file_structure with inline content."""

    def test_parse_gnn_file_extracts_structure_from_content(self) -> None:
        from gnn.processing import parse_gnn_file

        content = (
            "## ModelName\n\nDemo\n\n"
            "## StateSpaceBlock\n\nx: state\n\n"
            "## Connections\n\nx -> x\n"
        )
        result = parse_gnn_file(Path("inline.md"), content=content)
        assert result["success"] is True
        assert result["file_name"] == "inline.md"
        assert "ModelName" in result["sections"]
        assert "StateSpaceBlock" in result["sections"]
        assert "x" in result["variables"]
        assert result["structure_info"]["section_count"] == 3
        assert result["structure_info"]["has_variables"] is True

    def test_check_gnn_file_structure_flags_empty_and_unbalanced(self) -> None:
        from gnn.processing import check_gnn_file_structure

        empty = check_gnn_file_structure(Path("empty.md"), content="")
        assert empty["valid"] is False
        assert "File is empty" in empty["errors"]

        unbalanced = check_gnn_file_structure(
            Path("unbalanced.md"),
            content="## ModelName\n\nmatrix[3, 3\n",
        )
        assert unbalanced["valid"] is True
        assert "Unmatched brackets detected" in unbalanced["warnings"]
