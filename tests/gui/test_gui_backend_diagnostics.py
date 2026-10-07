"""Broken optional GUI imports remain visible in real headless artifacts."""

import builtins
import json
import logging
import sys
import types
from pathlib import Path

import pytest

from gnn.gui.backend import detect_gradio_backend
from gnn.gui.gui_3 import processor


@pytest.mark.parametrize("failure", ["missing", "incomplete"])
def test_headless_backend_receipt_retains_import_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    if failure == "missing":
        original_import = builtins.__import__

        def refuse_gradio(name, *args, **kwargs):
            if name == "gradio":
                raise ModuleNotFoundError("missing transitive GUI dependency")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse_gradio)
        expected = "ModuleNotFoundError: missing transitive GUI dependency"
    else:
        monkeypatch.setitem(sys.modules, "gradio", types.ModuleType("gradio"))
        expected = "AttributeError: gradio import does not expose Blocks"
    status = detect_gradio_backend()
    assert not status.available and status.reason == expected
    monkeypatch.setattr(processor, "_GUI_BACKEND", status.name)
    monkeypatch.setattr(processor, "_GUI_BACKEND_REASON", status.reason)
    target = tmp_path / "models"
    target.mkdir()
    (target / "model.md").write_text("## ModelName\nSensor\n", encoding="utf-8")
    output = tmp_path / "22_gui_output"
    assert processor.run_gui(target, output, logging.getLogger(__name__))
    receipt = json.loads((output / "design_studio_status.json").read_text())
    assert receipt["backend_reason"] == expected
    assert receipt["status"] == "static_headless_mode"
    assert receipt["launched"] is False
    assert Path(receipt["export_file"]).is_file()
