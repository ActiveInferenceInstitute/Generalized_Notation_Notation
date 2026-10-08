"""Saved-model export contracts read by independent JSON/XML/text consumers."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

from gnn.export import export_gnn_model


def saved_model(root: Path) -> dict:
    data = {
        "model_type": "gnn",
        "metadata": {
            "model_name": "LocalSensor",
            "note": "sensor μ < threshold & action",
        },
        "sections": {
            "StateSpaceBlock": ["s[2,type=float]", "o[2,type=float]"],
            "ModelParameters": ["threshold = 0.25"],
        },
        "variables": [{"name": "s", "type": "float"}, {"name": "o", "type": "float"}],
        "connections": [{"source": "s", "target": "o"}],
    }
    source = root / "saved-model.json"
    source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return json.loads(source.read_text(encoding="utf-8"))


def test_saved_model_semantics_survive_public_multiformat_export(
    tmp_path: Path,
) -> None:
    model = saved_model(tmp_path)
    output = tmp_path / "exports"
    output.mkdir()
    sentinel = output / "caller-owned.txt"
    sentinel.write_text("retain this neighbor")
    result = export_gnn_model(model, output, formats=["json", "xml", "txt", "dsl"])
    assert result["success"] is True
    assert result["errors"] == []
    assert set(result["exports"]) == {"json", "xml", "txt", "dsl"}
    paths = {name: Path(entry["file"]) for name, entry in result["exports"].items()}
    assert all(entry["success"] is True for entry in result["exports"].values())
    assert all(
        path.resolve().is_relative_to(output.resolve()) for path in paths.values()
    )
    assert json.loads(paths["json"].read_text(encoding="utf-8")) == model
    xml = ElementTree.parse(paths["xml"]).getroot()
    assert xml.findtext("metadata/note") == model["metadata"]["note"]
    assert [
        (node.get("name"), node.get("type"))
        for node in xml.findall("variables/variable")
    ] == [("s", "float"), ("o", "float")]
    assert [
        (node.get("source"), node.get("target"))
        for node in xml.findall("connections/connection")
    ] == [("s", "o")]
    assert {
        node.get("name"): [line.text for line in node.findall("line")]
        for node in xml.findall("sections/section")
    } == model["sections"]
    summary = paths["txt"].read_text()
    assert "Variables (2):" in summary and "Connections (1):" in summary
    assert "s -> o" in summary and "StateSpaceBlock: 2 lines" in summary
    dsl = paths["dsl"].read_text()
    assert "s: float" in dsl and "o: float" in dsl and "s -> o" in dsl
    assert "threshold = 0.25" in dsl
    assert sentinel.read_text() == "retain this neighbor"
    assert json.loads((tmp_path / "saved-model.json").read_text()) == model


def test_unknown_format_reports_failure_before_creating_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "uncreated"
    result = export_gnn_model(
        saved_model(tmp_path), output, formats=["unsupported-native-format"]
    )
    assert result["success"] is False
    assert result["exports"] == {}
    assert "Unsupported format" in result["error"]
    assert not output.exists()


def test_partial_format_failure_keeps_the_successful_saved_json_truthful(
    tmp_path: Path,
) -> None:
    output = tmp_path / "partial"
    output.mkdir()
    model = saved_model(tmp_path)
    result = export_gnn_model(
        model, output, formats=["json", "unsupported-native-format"]
    )
    assert result["success"] is False
    assert set(result["exports"]) == {"json"}
    assert result["exports"]["json"]["success"] is True
    assert json.loads(Path(result["exports"]["json"]["file"]).read_text()) == model
    assert result["errors"] and "Unsupported format" in result["error"]


def test_blocked_output_preserves_the_caller_file_and_reports_each_failure(
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "caller-file"
    blocked.write_bytes(b"immutable caller content")
    result = export_gnn_model(
        saved_model(tmp_path), blocked, formats=["json", "xml", "txt", "dsl"]
    )
    assert result["success"] is False
    assert result["error"]
    assert set(result["exports"]) == {"json", "xml", "txt", "dsl"}
    assert all(entry["success"] is False for entry in result["exports"].values())
    assert blocked.read_bytes() == b"immutable caller content"
