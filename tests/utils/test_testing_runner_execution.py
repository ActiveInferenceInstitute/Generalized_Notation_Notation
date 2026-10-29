"""Observed execution and refusal contracts for the public testing utilities.

All real pytest workers select temporary files, never this repository suite.
"""

from __future__ import annotations

import builtins
import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from gnn.utils.testing import reports, runner


def _test_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "test_selected.py"
    path.write_text(body)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    fast: fast selection\n    unit: unit selection\n"
    )
    return path


@pytest.mark.parametrize(
    ("body", "success", "passed"),
    [
        ("def test_pass(): assert True\n", True, 1),
        ("def test_fail(): assert False\n", False, 0),
        ("value = 1\n", False, 0),
        (
            (
                Path(__file__).resolve().parents[1]
                / "test_data/runner/all_skipped.py.txt"
            ).read_text(encoding="utf-8"),
            False,
            0,
        ),
        (
            "import pytest\ndef test_pass(): pass\n"
            "@pytest.fixture\ndef broken(): raise RuntimeError('setup failed')\n"
            "def test_setup_error(broken): pass\n",
            False,
            1,
        ),
    ],
)
def test_public_runner_uses_observed_subprocess_outcome(
    tmp_path: Path,
    body: str,
    success: bool,
    passed: int,
) -> None:
    path = _test_file(tmp_path, body)
    result = runner.TestRunner({"timeout_seconds": 20}).run_tests(
        [path],
        tmp_path / "results",
    )
    assert result["success"] is success
    assert result["tests_passed"] == passed
    assert (tmp_path / "results" / "pytest_stdout.txt").is_file()
    if "test_skipped" in body:
        assert result["tests_skipped"] == 1
        assert result["tests_run"] == 1
        assert "unverified" in result["error_message"]


def test_category_selects_actual_tests_and_unknown_category_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _test_file(
        tmp_path,
        "import pytest\n@pytest.mark.unit\ndef test_unit(): pass\n"
        "def test_other(): assert False\n",
    )
    monkeypatch.setattr(runner, "TEST_DIR", tmp_path)
    assert runner.run_test_category("unit", tmp_path, tmp_path / "results")
    assert not runner.run_test_category("unknown", tmp_path, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_mcp_report_has_only_current_executed_categories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _test_file(tmp_path, "import pytest\n@pytest.mark.fast\ndef test_fast(): pass\n")
    monkeypatch.setattr(runner, "TEST_DIR", tmp_path)
    monkeypatch.setitem(runner.TEST_CONFIG, "fast_only", True)
    monkeypatch.setitem(runner.TEST_CONFIG, "include_performance", True)
    result = reports.run_all_tests_mcp(str(tmp_path), str(tmp_path / "output"))
    assert result["success"] is True
    assert result["test_categories"] == ["fast"]
    assert set(result["results"]) == {"fast"}
    assert result["results"]["fast"] is True
    assert result["category_results"]["fast"]["tests_passed"] == 1
    report = tmp_path / "output" / "test_results" / "test_report.json"
    assert json.loads(report.read_text())["results"] == result["results"]

    assert reports.get_test_results(tmp_path / "output")["results"] == {"fast": True}
    _test_file(
        tmp_path, "import pytest\n@pytest.mark.fast\ndef test_fast(): assert False\n"
    )
    failed = reports.run_all_tests_mcp(str(tmp_path), str(tmp_path / "output"))
    assert failed["success"] is False
    assert failed["results"]["fast"] is False
    assert not bool(failed["results"]["fast"])
    assert not all(failed["results"].values())
    assert failed["category_results"]["fast"]["tests_failed"] == 1
    assert json.loads(report.read_text())["results"]["fast"] is False
    saved = reports.get_test_results(tmp_path / "output")
    assert saved["results"]["fast"] is False
    assert saved["category_results"]["fast"]["tests_failed"] == 1
    assert reports.get_test_results(report)["results"]["fast"] is False

    monkeypatch.setattr(runner, "TEST_DIR", tmp_path / "missing")
    unavailable = reports.run_all_tests_mcp(str(tmp_path), str(tmp_path / "output"))
    assert unavailable["results"]["fast"] is False
    assert unavailable["category_results"]["fast"]["tests_run"] == 0


def test_missing_repository_harness_cannot_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def import_without_harness(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("tests.infrastructure"):
            raise ImportError("repository harness absent")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_harness)
    result = runner.TestRunner().run_tests([tmp_path], tmp_path / "unused")
    assert result["success"] is False
    assert "repository harness absent" in result["error"]


def test_deadline_bounds_real_pytest_worker(tmp_path: Path) -> None:
    path = _test_file(tmp_path, "import time\ndef test_wait(): time.sleep(30)\n")
    started = time.monotonic()
    result = runner.TestRunner({"timeout_seconds": 0.2}).run_tests(
        [path],
        tmp_path / "results",
    )
    assert result["success"] is False
    assert "timed out" in result["error_message"]
    assert time.monotonic() - started < 10


def test_cancellation_stops_runner_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.infrastructure.test_config import TestExecutionConfig
    from tests.infrastructure.test_runner import TestRunner as SourceTestRunner

    source = SourceTestRunner(TestExecutionConfig(parallel=False, coverage=False))
    stopped = []
    monkeypatch.setattr(source.resource_monitor, "start_monitoring", lambda: None)
    monkeypatch.setattr(
        source.resource_monitor, "stop_monitoring", lambda: stopped.append(True)
    )

    def cancel(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(source, "_execute_pytest", cancel)
    with pytest.raises(KeyboardInterrupt):
        source.run_tests([tmp_path], tmp_path)
    assert stopped


@pytest.mark.parametrize(
    "options",
    [
        {"parallel": True},
        {"timeout_seconds": float("nan")},
        {"max_failures": False},
        {"markers": "unit"},
    ],
)
def test_invalid_runner_controls_refuse_before_execution(
    tmp_path: Path,
    options: dict[str, Any],
) -> None:
    result = runner.TestRunner(options).run_tests([tmp_path], tmp_path / "unused")
    assert result["success"] is False
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("saved_result", [False, {"success": False, "tests_run": 0}])
def test_report_renderers_do_not_turn_failed_results_into_passes(
    tmp_path: Path,
    saved_result: Any,
) -> None:
    data = {"success": True, "results": {"unit": saved_result}}
    html = tmp_path / "report.html"
    markdown = tmp_path / "report.md"
    assert reports.generate_html_report_file(data, html)
    assert reports.generate_markdown_report_file(data, markdown)
    assert "unit: Failed" in html.read_text()
    assert "Failure" in html.read_text()
    assert "Passed" not in html.read_text()
    assert "Failed" in markdown.read_text()
    assert "Passed" not in markdown.read_text()


def test_comprehensive_test_report_requires_and_retains_saved_evidence(
    tmp_path: Path,
) -> None:
    logger = logging.getLogger("saved-test-report")
    output = tmp_path / "rendered"
    assert not reports.generate_comprehensive_report(tmp_path, output, logger)
    assert not (output / "test_reports" / "test_report.json").exists()
    source = tmp_path / "test_results" / "test_report.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "success": True,
                "results": {"unit": {"success": False, "tests_failed": 1}},
            }
        )
    )
    assert reports.generate_comprehensive_report(tmp_path, output, logger)
    retained = json.loads((output / "test_reports" / "test_report.json").read_text())
    assert retained["success"] is False
    assert retained["results"] == {"unit": False}
    assert retained["category_results"] == {
        "unit": {"success": False, "tests_failed": 1}
    }
    assert retained["source_report"] == str(source)


def test_stage_parallel_request_is_refused_and_serial_override_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _test_file(tmp_path, "import pytest\n@pytest.mark.fast\ndef test_fast(): pass\n")
    monkeypatch.setattr(runner, "TEST_DIR", tmp_path)
    assert runner.TEST_STAGES["fast"]["parallel"] is True
    assert not runner.run_test_stage("fast", tmp_path, tmp_path / "parallel")
    assert "requires serial test execution" in caplog.text
    assert not (tmp_path / "parallel").exists()
    monkeypatch.setitem(
        runner.TEST_STAGES,
        "fast",
        {
            **runner.TEST_STAGES["fast"],
            "parallel": False,
        },
    )
    assert runner.run_test_stage("fast", tmp_path, tmp_path / "serial")
    assert not runner.run_test_stage("unknown", tmp_path, tmp_path / "unknown")


def test_configured_fixture_path_resolves_outside_repository_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.utils.testing.constants import TEST_CONFIG, TEST_DIR
    from gnn.utils.testing.fixtures import get_sample_pipeline_arguments
    from tests.helpers import get_test_data_dir

    monkeypatch.chdir(tmp_path)
    configured = TEST_CONFIG["test_data_dir"]
    assert isinstance(configured, str)
    assert Path(configured) == TEST_DIR / "test_data"
    assert Path(configured) == get_test_data_dir()
    assert Path(configured).is_dir()
    assert get_sample_pipeline_arguments()["test_data_dir"] == configured


def test_foreign_test_harness_module_cannot_establish_gnn_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import ModuleType

    foreign = ModuleType("tests.infrastructure.test_config")
    foreign.__file__ = str(tmp_path / "foreign_config.py")
    monkeypatch.setitem(sys.modules, foreign.__name__, foreign)
    result = runner.TestRunner().run_tests([tmp_path], tmp_path / "unused")
    assert result["success"] is False
    assert "harness owner mismatch" in result["error"]
    assert not (tmp_path / "unused").exists()


def test_absent_owned_harness_files_refuse_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_HARNESS_DIR", tmp_path / "missing")
    result = runner.TestRunner().run_tests([tmp_path], tmp_path / "unused")
    assert result["success"] is False
    assert "harness files are unavailable" in result["error"]
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("category", [False, {"success": False, "tests_failed": 1}])
def test_saved_report_readback_keeps_boolean_statuses(
    tmp_path: Path,
    category: Any,
) -> None:
    source = tmp_path / "test_results.json"
    source.write_text(json.dumps({"success": False, "results": {"fast": category}}))
    saved = reports.get_test_results(tmp_path)
    assert saved["results"]["fast"] is False
    assert not all(saved["results"].values())
    if isinstance(category, dict):
        assert saved["category_results"]["fast"]["tests_failed"] == 1


def test_saved_overall_failure_survives_passing_categories_and_render(
    tmp_path: Path,
) -> None:
    source = tmp_path / "test_results" / "test_report.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "success": False,
                "cleanup_error": "cleanup unverified",
                "results": {"fast": True},
                "category_results": {"fast": {"success": True, "tests_passed": 1}},
            }
        )
    )
    saved = reports.get_test_results(tmp_path)
    assert saved["success"] is False
    assert saved["declared_success"] is False
    assert saved["results"] == {"fast": True}
    assert saved["cleanup_error"] == "cleanup unverified"
    assert saved["source_report"] == str(source)
    assert saved["readback_only"] is True
    output = tmp_path / "rendered"
    assert reports.generate_comprehensive_report(
        tmp_path, output, logging.getLogger("readback")
    )
    rendered = json.loads((output / "test_reports" / "test_report.json").read_text())
    assert rendered["success"] is False
    assert rendered["results"] == {"fast": True}
    assert rendered["cleanup_error"] == "cleanup unverified"
    assert rendered["source_report"] == str(source)
    assert rendered["readback_only"] is True


def test_ambiguous_saved_views_require_explicit_file(tmp_path: Path) -> None:
    stale = tmp_path / "test_results.json"
    stale.write_text(json.dumps({"success": True, "results": {"fast": True}}))
    current = tmp_path / "test_results" / "test_report.json"
    current.parent.mkdir()
    current.write_text(json.dumps({"success": False, "results": {"fast": False}}))
    ambiguous = reports.get_test_results(tmp_path)
    assert ambiguous["status"] == "ambiguous_results"
    assert ambiguous["success"] is False
    assert set(ambiguous["candidates"]) == {str(stale), str(current)}
    explicit = reports.get_test_results(current)
    assert explicit["success"] is False
    assert explicit["results"]["fast"] is False
    assert explicit["source_report"] == str(current)
    assert explicit["readback_only"] is True
    logger = logging.getLogger("ambiguous-readback")
    assert not reports.generate_comprehensive_report(
        tmp_path, tmp_path / "refused", logger
    )
    assert reports.generate_comprehensive_report(current, tmp_path / "explicit", logger)
    rendered = json.loads(
        (tmp_path / "explicit" / "test_reports" / "test_report.json").read_text()
    )
    assert rendered["success"] is False
    assert rendered["source_report"] == str(current)


def test_cleanup_failure_is_published_after_real_category_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _test_file(tmp_path, "import pytest\n@pytest.mark.fast\ndef test_fast(): pass\n")
    monkeypatch.setattr(runner, "TEST_DIR", tmp_path)
    monkeypatch.setitem(runner.TEST_CONFIG, "fast_only", True)

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup unverified")

    monkeypatch.setattr(runner, "cleanup_test_environment", fail_cleanup)
    result = reports.run_all_tests_mcp(str(tmp_path), str(tmp_path / "output"))
    assert result["success"] is False
    assert result["results"]["fast"] is True
    assert result["category_results"]["fast"]["tests_passed"] == 1
    assert result["cleanup_error"] == "cleanup unverified"
    saved = reports.get_test_results(tmp_path / "output")
    assert saved["success"] is False
    assert saved["results"]["fast"] is True
    assert saved["cleanup_error"] == "cleanup unverified"
