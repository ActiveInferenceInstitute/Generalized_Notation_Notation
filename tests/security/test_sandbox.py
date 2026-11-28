#!/usr/bin/env python3
"""Tests for the opt-in Step-12 sandbox wrapper (execute.sandbox)."""

import logging
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute import sandbox
from gnn.execute.processor import execute_single_script
from gnn.execute.subprocess_envelope import NEVER_STARTED, run_subprocess_envelope


class TestSandbox:
    """Sandbox wrapper behavior (mode resolution, wrapping, execution)."""

    def test_detect_sandbox_returns_spec_or_none(self) -> None:
        spec = sandbox.detect_sandbox()
        assert spec is None or isinstance(spec, sandbox.SandboxSpec)

    def test_wrap_command_prepends_prefix(self) -> None:
        spec = sandbox.SandboxSpec("firejail", ("firejail", "--net=none"))
        wrapped = sandbox.wrap_command(["python3", "x.py"], spec)
        assert wrapped == ["firejail", "--net=none", "python3", "x.py"]

    def test_run_sandboxed_off_executes_unsandboxed(self) -> None:
        result = sandbox.run_sandboxed(["python3", "-c", "print(42)"], mode="off")
        assert result["success"] is True
        assert result["sandboxed"] is False
        assert result["return_code"] == 0

    def test_run_sandboxed_require_blocks_without_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: None)
        result = sandbox.run_sandboxed(["python3", "-c", "print(1)"], mode="require")
        assert result["blocked"] is True
        assert result["sandboxed"] is False
        assert result["success"] is False
        assert result["error_type"] == "SandboxUnavailable"

    def test_run_sandboxed_prefer_falls_back_without_backend(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: None)
        result = sandbox.run_sandboxed(["python3", "-c", "print(1)"], mode="prefer")
        assert result["blocked"] is False
        assert result["sandboxed"] is False
        assert result["success"] is True
        receipts = [record for record in caplog.records if "sandbox_" in record.message]
        assert len(receipts) == 1
        assert "sandbox_disabled_receipt" in receipts[0].message
        assert "GNN_SANDBOX=prefer, no backend found" in receipts[0].message

    def test_run_sandboxed_uses_backend_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = sandbox.SandboxSpec("firejail", ("echo", "SAND"))
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: spec)
        result = sandbox.run_sandboxed(["python3", "-c", "print(1)"], mode="require")
        assert result["sandboxed"] is True
        assert result["sandbox"] == "firejail"

    @pytest.mark.parametrize(
        ("setting", "mode", "error_type"),
        [
            ("", "invalid", "InvalidSandboxMode"),
            ("requre", "invalid", "InvalidSandboxMode"),
            ("require", "require", "SandboxUnavailable"),
        ],
    )
    def test_refused_policy_blocks_all_launch_surfaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        setting: str,
        mode: str,
        error_type: str,
    ) -> None:
        monkeypatch.setenv("GNN_SANDBOX", setting)
        monkeypatch.setattr(sandbox, "detect_sandbox", lambda: None)

        def forbid_launch(*args: object, **kwargs: object) -> None:
            pytest.fail("A blocked sandbox policy must prevent launch")

        monkeypatch.setattr(subprocess, "Popen", forbid_launch)
        monkeypatch.setattr(
            "gnn.execute.processor._check_framework_by_name", forbid_launch
        )
        script = tmp_path / "model" / "pymdp" / "rendered.py"
        script.parent.mkdir(parents=True)
        script.write_text("print('test model')\n")
        command = [sys.executable, str(script)]

        direct = sandbox.run_sandboxed(command)
        assert direct["blocked"] is True
        assert direct["mode"] == mode

        envelope = run_subprocess_envelope(command)
        assert envelope["containment"] == "not_started"
        assert envelope["sandbox_mode"] == mode

        step = execute_single_script(
            {
                "path": script,
                "framework": "pymdp",
                "name": script.name,
                "executor": sys.executable,
            },
            tmp_path / "results",
            False,
            logging.getLogger(__name__),
        )
        assert step["execution_metadata"]["sandbox_mode"] == mode
        for result in (direct, envelope, step):
            assert result["error_type"] == error_type
            assert result["success"] is False
            assert result["return_code"] == NEVER_STARTED
