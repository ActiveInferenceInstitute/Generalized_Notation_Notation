#!/usr/bin/env python3
"""Behavior tests for the shared utils helpers (consolidated 2026-09-04).

Pins the single-source-of-truth implementations introduced when duplicated
logic across src/gnn/utils/ was collapsed:

- ``io_utils.verify_directory_writable`` — the one writable-probe behind
  ``gnn.utils.pipeline_orchestration.pipeline_validator.check_pipeline_readiness``
- the canonical memory probe ``gnn.utils.runtime_safety.resource_manager.get_memory_usage``
- ``resource_manager.with_resource_limits`` exception-propagation semantics
- the shared fallback-default table behind ``ArgumentParser``
- ``StepConfiguration.validate_step_args`` injectable ``project_root``
- ``pipeline_monitor`` duration-variance alert bands (``critical`` key)
- ``mcp`` environment redaction helpers

All tests are deterministic, isolated, and network-free.
"""

from __future__ import annotations

import argparse
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.utils.arguments.arg_parsing import ArgumentParser, fallback_default_for
from gnn.utils.arguments.step_config import StepConfiguration
from gnn.utils.config_io.io_utils import verify_directory_writable
from gnn.utils.mcp.server import is_sensitive_env_key, redact_environment
from gnn.utils.pipeline_orchestration.pipeline_monitor import (
    AlertLevel,
    PipelineMonitor,
)
from gnn.utils.runtime_safety.resource_manager import (
    get_memory_usage,
    with_resource_limits,
)


def _make_readonly(path: Path) -> None:
    """Make *path* read-only for the current user (POSIX)."""
    path.chmod(path.stat().st_mode & ~stat.S_IWUSR)


def _restore_permissions(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IWUSR)


requires_write_permissions = pytest.mark.needs_nonroot


class TestVerifyDirectoryWritable:
    """The shared create-rename-cleanup probe."""

    def test_roundtrip_leaves_directory_clean(self, tmp_path: Path) -> None:
        verify_directory_writable(tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_custom_probe_name(self, tmp_path: Path) -> None:
        verify_directory_writable(tmp_path, probe_name="step_test.tmp")
        assert list(tmp_path.iterdir()) == []

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            verify_directory_writable(tmp_path / "does_not_exist")

    def test_file_target_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "a_file.txt"
        target.write_text("not a directory")
        with pytest.raises(OSError):
            verify_directory_writable(target)

    @requires_write_permissions
    def test_readonly_directory_raises(self, tmp_path: Path) -> None:
        _make_readonly(tmp_path)
        try:
            with pytest.raises(OSError):
                verify_directory_writable(tmp_path)
        finally:
            _restore_permissions(tmp_path)


class TestSharedProbeCallers:
    """The remaining former probe copy must behave identically through the helper."""

    def test_readiness_ready_when_output_writable(self, tmp_path: Path) -> None:
        from gnn.utils.pipeline_orchestration.pipeline_validator import (
            check_pipeline_readiness,
        )

        args = SimpleNamespace(target_dir=tmp_path, output_dir=tmp_path / "out")
        result = check_pipeline_readiness([("2_tests.py", "tests")], args)
        assert result["ready"] is True
        assert result["blocking_issues"] == []
        assert (tmp_path / "out").is_dir()

    @requires_write_permissions
    def test_readiness_blocks_when_output_unwritable(self, tmp_path: Path) -> None:
        from gnn.utils.pipeline_orchestration.pipeline_validator import (
            check_pipeline_readiness,
        )

        out_dir = tmp_path / "locked"
        out_dir.mkdir()
        args = SimpleNamespace(target_dir=tmp_path, output_dir=out_dir)
        _make_readonly(out_dir)
        try:
            result = check_pipeline_readiness([("2_tests.py", "tests")], args)
        finally:
            _restore_permissions(out_dir)
        assert result["ready"] is False
        assert any("not writable" in issue for issue in result["blocking_issues"])


class TestCanonicalMemoryProbe:
    """One psutil-backed probe; the other modules re-export it."""

    def test_resource_manager_alias(self) -> None:
        import gnn.utils.runtime_safety.resource_manager as rm

        assert rm.get_memory_usage is rm.get_current_memory_usage

    def test_probe_returns_non_negative_float(self) -> None:
        value = get_memory_usage()
        assert isinstance(value, float)
        assert value >= 0.0


class TestWithResourceLimits:
    """Body exceptions must never be masked by the guard."""

    def test_limit_violation_raises_when_body_succeeds(self) -> None:
        with pytest.raises(RuntimeError, match="Time limit exceeded"):
            with with_resource_limits(max_time_seconds=-1.0):
                pass

    def test_body_exception_propagates_when_limits_ok(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            with with_resource_limits():
                raise ValueError("boom")

    def test_body_exception_wins_over_limit_violation(self) -> None:
        with pytest.raises(ValueError, match="boom"):
            with with_resource_limits(max_time_seconds=-1.0):
                raise ValueError("boom")


class TestFallbackDefaults:
    """The shared recovery-default table (replaced two drifted ladders)."""

    @pytest.mark.parametrize(
        ("arg_name", "expected"),
        [
            ("recursive", True),
            ("verbose", False),
            ("strict", False),
            ("estimate_resources", True),
            ("fast_only", True),
            ("comprehensive", False),
            ("advanced_stats", False),
            ("generate_animations", True),
            ("execution_summary_detail", False),
            ("llm_timeout", None),
            ("bottleneck_threshold", 60.0),
            ("duration", 30.0),
            ("timeout", 300),
            ("execution_workers", 1),
            ("simulation_params", "{}"),
            ("viz_type", "all"),
            ("gui_types", "gui_1,gui_2"),
            ("timesteps", None),
            ("optional_groups", None),
            ("analysis_model", None),
            ("unknown_argument", None),
            ("output_dir", Path("output")),
            ("render_output_dir", Path("output")),
            ("target_dir", Path("input/gnn_files")),
            ("ontology_terms_file", None),
        ],
    )
    def test_defaults(self, arg_name: str, expected: Any) -> None:
        assert fallback_default_for(arg_name) == expected

    def test_mutable_default_is_copied(self) -> None:
        first = fallback_default_for("export_formats")
        assert first == ["html", "json"]
        first.append("xml")
        assert fallback_default_for("export_formats") == ["html", "json"]

    def test_create_default_namespace_matches_contract(self) -> None:
        ns_render = ArgumentParser.create_default_namespace("11_render.py")
        assert ns_render.timesteps is None
        assert ns_render.simulation_params == "{}"

        ns_analysis = ArgumentParser.create_default_namespace("16_analysis.py")
        assert ns_analysis.advanced_stats is False
        assert ns_analysis.generate_animations is True

    def test_parse_step_arguments_guarantees_all_step_attrs(self) -> None:
        args = ArgumentParser.parse_step_arguments("17_integration.py", [])
        for arg_name in ArgumentParser.STEP_ARGUMENTS["17_integration.py"]:
            assert hasattr(args, arg_name), arg_name
        assert args.recursive is True
        assert args.verbose is False
        assert args.target_dir == Path("input/gnn_files")
        assert args.output_dir == Path("output")


class TestValidateStepArgsProjectRoot:
    """Injectable project root (existing frame heuristic still the default)."""

    def test_unknown_step(self) -> None:
        errors = StepConfiguration.validate_step_args(
            "99_missing", argparse.Namespace()
        )
        assert errors == ["Unknown step: 99_missing"]

    def test_missing_required_arg_reported(self) -> None:
        errors = StepConfiguration.validate_step_args("3_gnn", argparse.Namespace())
        assert any("Missing required argument" in e for e in errors)

    def test_project_root_repairs_missing_input_path(self, tmp_path: Path) -> None:
        # A target_dir that does not exist as given (nor relative to CWD),
        # whose *name* exists directly under the injected project root.
        (tmp_path / "gnn_files_fixture").mkdir()
        missing = Path("input/gnn_files_fixture")
        assert not missing.exists()
        args = argparse.Namespace(target_dir=missing, output_dir=Path("output"))
        errors = StepConfiguration.validate_step_args(
            "3_gnn", args, project_root=tmp_path
        )
        assert not any("Path does not exist" in e for e in errors)
        assert args.target_dir == tmp_path / "gnn_files_fixture"

    def test_project_root_reports_unresolvable(self, tmp_path: Path) -> None:
        missing = Path("input/definitely_missing_gnn_files_fixture")
        assert not missing.exists()
        args = argparse.Namespace(target_dir=missing, output_dir=Path("output"))
        errors = StepConfiguration.validate_step_args(
            "3_gnn", args, project_root=tmp_path
        )
        assert any("Path does not exist" in e for e in errors)
        # Unresolvable paths are reported, never silently rewritten.
        assert args.target_dir == missing


class TestMonitorAlertBands:
    """Duration-variance thresholds: healthy <=1.5x, degraded <=2x, <=3x, critical >3x."""

    def _monitor_with_capture(self) -> tuple[PipelineMonitor, List[AlertLevel]]:
        captured: List[AlertLevel] = []
        monitor = PipelineMonitor(
            alert_callbacks=[lambda alert: captured.append(alert.level)]
        )
        return monitor, captured

    def test_critical_threshold_exists(self) -> None:
        monitor = PipelineMonitor()
        assert monitor.health_thresholds["duration_variance"]["critical"] == 3.0

    def test_beyond_critical_band_fires_critical(self) -> None:
        monitor, captured = self._monitor_with_capture()
        monitor.performance_baselines["step"] = 1.0
        monitor._check_performance_alerts("step", 5.0)
        assert captured == [AlertLevel.CRITICAL]

    def test_degraded_band_fires_warning(self) -> None:
        monitor, captured = self._monitor_with_capture()
        monitor.performance_baselines["step"] = 1.0
        monitor._check_performance_alerts("step", 2.5)
        assert captured == [AlertLevel.WARNING]

    def test_within_baseline_fires_nothing(self) -> None:
        monitor, captured = self._monitor_with_capture()
        monitor.performance_baselines["step"] = 1.0
        monitor._check_performance_alerts("step", 1.2)
        assert captured == []


class TestMcpRedaction:
    """Environment redaction used by the MCP info tools."""

    @pytest.mark.parametrize(
        "key",
        [
            "AWS_SESSION_TOKEN",
            "GOOGLE_API_KEY",
            "DATABASE_CREDENTIALS",
            "MY_PASSWD",
            "BASIC_AUTH",
            "OPENAI_SECRET_KEY",
        ],
    )
    def test_sensitive_keys_detected(self, key: str) -> None:
        assert is_sensitive_env_key(key) is True

    @pytest.mark.parametrize("key", ["HOME", "MPLBACKEND", "LANG", "CI"])
    def test_safe_keys_pass(self, key: str) -> None:
        assert is_sensitive_env_key(key) is False

    def test_redact_environment_drops_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("UTILS_TEST_SECRET_TOKEN", "hunter2")
        monkeypatch.setenv("UTILS_TEST_SAFE", "visible")
        redacted = redact_environment()
        assert "UTILS_TEST_SECRET_TOKEN" not in redacted
        assert redacted.get("UTILS_TEST_SAFE") == "visible"
