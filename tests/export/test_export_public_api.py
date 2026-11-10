"""Tests for export module's public API surface not covered by existing tests.

Covers: export_to_json_gnn, export_to_xml_gnn, export_to_plaintext_summary,
export_to_plaintext_dsl, export_to_python_pickle, export_model,
export_single_gnn_file, parse_gnn_content, generate_exports,
get_supported_formats_dict, _gnn_model_to_dict, HAS_NETWORKX, FEATURES.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestExportConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import gnn.export as export

        assert hasattr(export, "FEATURES")
        assert isinstance(export.FEATURES, dict)
        assert export.FEATURES.get("json_export") is True

    def test_has_networkx_flag(self) -> None:
        import gnn.export as export

        assert hasattr(export, "HAS_NETWORKX")
        assert isinstance(export.HAS_NETWORKX, bool)

    def test_get_supported_formats_dict(self) -> None:
        from gnn.export import get_supported_formats_dict

        result = get_supported_formats_dict()
        assert isinstance(result, dict)
        assert "data_formats" in result
        assert "graph_formats" in result
        assert "text_formats" in result


class TestExportFormatters:
    """Test individual format export functions."""

    def test_export_to_json_gnn(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_json_gnn

        out = tmp_path / "out.json"
        result = export_to_json_gnn({"name": "test", "variables": []}, out)
        assert result is True
        assert out.exists()

    def test_export_to_xml_gnn(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_xml_gnn

        out = tmp_path / "out.xml"
        result = export_to_xml_gnn({"name": "test"}, out)
        assert result is True
        assert out.exists()

    def test_export_to_plaintext_summary(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_plaintext_summary

        out = tmp_path / "out.txt"
        data: dict[str, Any] = {"name": "TestModel", "variables": [{"name": "s"}]}
        result = export_to_plaintext_summary(data, out)
        assert result is True
        assert out.exists()
        content = out.read_text()
        # Summary uses "Model Type: gnn" or similar header
        assert "GNN Model Summary" in content or "gnn" in content

    def test_export_to_plaintext_dsl(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_plaintext_dsl

        out = tmp_path / "out.gnn"
        result = export_to_plaintext_dsl(
            {"name": "M", "variables": [{"name": "s"}]}, out
        )
        assert result is True
        assert out.exists()

    def test_export_to_python_pickle(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_python_pickle

        out = tmp_path / "model.pkl"
        result = export_to_python_pickle({"key": "value"}, out)
        assert result is True
        assert out.exists()

    def test_export_to_graphml(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_graphml

        out = tmp_path / "model.graphml"
        data: dict[str, Any] = {
            "name": "M",
            "variables": [{"name": "s"}, {"name": "o"}],
            "connections": [{"source": "s", "target": "o", "directed": True}],
        }
        result = export_to_graphml(data, out)
        assert result is True

    def test_export_to_gexf(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_gexf

        out = tmp_path / "model.gexf"
        data: dict[str, Any] = {
            "name": "M",
            "nodes": [{"id": "s"}, {"id": "o"}],
            "edges": [{"source": "s", "target": "o"}],
        }
        result = export_to_gexf(data, out)
        assert result is True

    def test_export_to_pickle(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_pickle

        out = tmp_path / "data.pkl"
        result = export_to_pickle([1, 2, 3], out)
        assert result is True
        assert out.exists()

    def test_export_to_xml(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_xml

        out = tmp_path / "data.xml"
        result = export_to_xml({"item": "value"}, out)
        assert result is True
        assert out.exists()

    def test_export_to_json(self, tmp_path: Any) -> None:
        from gnn.export.formatters import export_to_json

        out = tmp_path / "data.json"
        result = export_to_json({"key": "val"}, out)
        assert result is True
        assert out.exists()


class TestExportProcessor:
    """Test processor-level functions."""

    def test_gnn_model_to_dict(self) -> None:
        from gnn.export.processor import _gnn_model_to_dict

        result = _gnn_model_to_dict("# Test\n## ModelName\nM")
        assert isinstance(result, dict)

    def test_gnn_model_to_dict_empty(self) -> None:
        from gnn.export.processor import _gnn_model_to_dict

        result = _gnn_model_to_dict("")
        assert isinstance(result, dict)

    def test_parse_gnn_content(self) -> None:
        from gnn.export import parse_gnn_content

        result = parse_gnn_content("# Test\n## ModelName\nM\n## StateSpaceBlock\ns[3]")
        assert isinstance(result, dict)

    def test_parse_gnn_content_empty(self) -> None:
        from gnn.export import parse_gnn_content

        result = parse_gnn_content("")
        assert isinstance(result, dict)

    def test_export_single_gnn_file(self, tmp_path: Any) -> None:
        from gnn.export import export_single_gnn_file

        gnn_file = tmp_path / "test.md"
        gnn_file.write_text("# Test\n## ModelName\nM\n## StateSpaceBlock\ns[3]")
        out_dir = tmp_path / "out"
        result = export_single_gnn_file(gnn_file, out_dir)
        assert result is not None

    def test_export_model_single_format(self, tmp_path: Any) -> None:
        from gnn.export import export_model

        data: dict[str, Any] = {"name": "M", "variables": []}
        result = export_model(data, tmp_path, formats=["json"])
        assert isinstance(result, dict)

    def test_export_model_multiple_formats(self, tmp_path: Any) -> None:
        from gnn.export import export_model

        data: dict[str, Any] = {"name": "M", "variables": [{"name": "s"}]}
        result = export_model(data, tmp_path, formats=["json", "xml", "unknown"])
        assert result["success"] is False
        assert result["errors"] == ["Unsupported format: unknown"]
        assert result["formats"] == {"json": True, "xml": True}
        assert (tmp_path / "model.json").is_file()
        assert (tmp_path / "model.xml").is_file()

        unsupported = export_model(data, tmp_path / "unsupported", formats=["unknown"])
        assert unsupported["success"] is False
        assert unsupported["errors"] == ["Unsupported format: unknown"]
        assert unsupported["exports"] == {}
        assert not any((tmp_path / "unsupported").iterdir())

    def test_generate_exports_empty_dir(self, tmp_path: Any) -> None:
        from gnn.export import generate_exports

        target = tmp_path / "empty"
        target.mkdir()
        out = tmp_path / "out"
        result = generate_exports(target_dir=target, output_dir=out)
        assert isinstance(result, bool)
