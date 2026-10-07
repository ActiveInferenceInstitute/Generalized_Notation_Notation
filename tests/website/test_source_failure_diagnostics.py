"""Saved sites retain useful source-read causes and their usable placeholders."""

from pathlib import Path

import pytest

from gnn.website import WebsiteGenerator


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_unreadable_source_keeps_listing_and_model_pages(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    source = tmp_path / "unreadable.md"
    if kind == "directory":
        source.mkdir()
    output = tmp_path / "site"
    result = WebsiteGenerator().generate_website(
        {
            "output_dir": str(output),
            "gnn_files": [source],
            "models": [
                {"name": "Sensor", "source": source, "source_name": source.name}
            ],
        },
        filesystem=False,
    )
    assert result["success"] and result["pages_created"] == 7
    assert result["model_pages"] == ["model/sensor.html"]
    assert "(could not read)" in (output / "gnn_files.html").read_text()
    assert "(could not read source file)" in (output / "model/sensor.html").read_text()
    expected = "FileNotFoundError" if kind == "missing" else "IsADirectoryError"
    assert expected in caplog.text and str(source) in caplog.text
    assert "Could not read website source" in caplog.text
    assert "Could not read model source" in caplog.text
