"""Real read/write failures keep the runner's verdict and contextual cause."""

from pathlib import Path

import pytest

from gnn.execute.rxinfer import rxinfer_runner


def test_invalid_script_encoding_refuses_before_process_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    script = tmp_path / "invalid_rxinfer.jl"
    script.write_bytes(b"\xff\xfe")

    def forbidden_launch(*args, **kwargs):
        pytest.fail("Unreadable source must not reach the subprocess boundary")

    monkeypatch.setattr(rxinfer_runner, "run_subprocess_envelope", forbidden_launch)
    assert rxinfer_runner.execute_rxinfer_script(script) is False
    assert str(script) in caplog.text
    assert "UnicodeDecodeError" in caplog.text


@pytest.mark.parametrize("success", [False, True])
def test_blocked_evidence_path_keeps_actual_run_verdict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    success: bool,
) -> None:
    script = tmp_path / "source_rxinfer.jl"
    script.write_text('println("native boundary")\n', encoding="utf-8")
    blocked = tmp_path / "occupied"
    blocked.write_text("regular file", encoding="utf-8")
    envelope = {
        "success": success,
        "return_code": 0 if success else 1,
        "stdout": "captured stdout",
        "stderr": "" if success else "native error",
        "duration_seconds": 0.1,
    }
    monkeypatch.setattr(
        rxinfer_runner, "run_subprocess_envelope", lambda *a, **k: envelope
    )
    assert rxinfer_runner.execute_rxinfer_script(script, output_dir=blocked) is success
    assert str(script) in caplog.text and str(blocked) in caplog.text
    assert "FileExistsError" in caplog.text
    assert blocked.read_text() == "regular file"
