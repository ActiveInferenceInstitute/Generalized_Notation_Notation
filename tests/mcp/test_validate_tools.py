"""Unit tests for ``gnn.mcp.validate_tools`` (the audit script's ``main()``).

``main()`` previously had zero test coverage: the MCP init-failure exit, the
per-tool ``NOT_CALLABLE`` / ``UNDOCUMENTED`` classification, the callability
spot-check branches (SKIP / ``success=False`` / exception), and the
``logging_miss`` ``register_tools`` scan were all unpinned. These tests
isolate ``main()`` from the live MCP registry by stubbing
``gnn.mcp.initialize`` / ``gnn.mcp.mcp_instance`` and pointing ``SRC_ROOT``
at a tmp tree so the audit report is written to tmp instead of the repo.

Deliberately unmarked: the audit is offline logic exercised by the default
suite (the ``mcp`` marker would deselect it locally).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gnn.mcp import validate_tools  # noqa: E402


def _install_fake_mcp(
    monkeypatch: pytest.MonkeyPatch,
    tools: dict[str, Any],
    modules: dict[str, Any] | None = None,
    execute_tool_fn: Callable[[str], Any] | None = None,
) -> SimpleNamespace:
    """Stub ``gnn.mcp.initialize``/``mcp_instance`` for one ``main()`` call."""

    def _default_execute_tool(name: str, args: dict[str, Any]) -> Any:
        return {"success": False, "error": "unconfigured"}

    fake = SimpleNamespace(
        tools=tools,
        modules=modules or {},
        execute_tool=execute_tool_fn or _default_execute_tool,
    )

    def initialize(**kwargs: Any) -> None:
        return None

    monkeypatch.setattr("gnn.mcp.initialize", initialize)
    monkeypatch.setattr("gnn.mcp.mcp_instance", fake)
    return fake


def _make_tool(
    *,
    func: Any = None,
    description: str = "",
    module: str = "tests",
    category: str = "test",
) -> Any:
    return SimpleNamespace(
        func=func,
        function=None,
        description=description,
        module=module,
        category=category,
    )


def _real_func() -> dict[str, Any]:
    return {"success": True}


def _seed_src_tree(
    tmp_path: Path, good_modules: list[str], bad_modules: list[str]
) -> Path:
    """Create the ``<SRC_ROOT>/gnn/<mod>/mcp.py`` tree the logging scan reads."""
    src_root = tmp_path / "src"
    (src_root / "gnn" / "mcp").mkdir(parents=True)
    for name in good_modules:
        d = src_root / "gnn" / name
        d.mkdir()
        (d / "mcp.py").write_text(
            "def register_tools(mcp):\n    logger.info('registered')\n",
            encoding="utf-8",
        )
    for name in bad_modules:
        d = src_root / "gnn" / name
        d.mkdir()
        (d / "mcp.py").write_text(
            "def register_tools(mcp):\n    pass\n", encoding="utf-8"
        )
    return src_root


def _read_report(src_root: Path) -> dict[str, Any]:
    report_path = src_root / "gnn" / "mcp" / "audit_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


@pytest.mark.unit
class TestValidateToolsMain:
    def test_clean_audit_returns_zero_and_writes_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=["alpha_mod"], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "tool_a": _make_tool(func=_real_func, description="Does A"),
            "list_analysis_tools": _make_tool(
                func=_real_func, description="Lists analysis tools"
            ),
            "list_render_frameworks": _make_tool(
                func=_real_func, description="Lists render frameworks"
            ),
        }
        _install_fake_mcp(
            monkeypatch,
            tools,
            modules={"alpha_mod": SimpleNamespace(status="loaded", tools_count=3)},
            execute_tool_fn=lambda name, args: {"success": True},
        )

        rc = validate_tools.main()

        assert rc == 0
        report = _read_report(src_root)
        assert report["tools_total"] == 3
        assert report["modules_loaded"] == 1
        assert report["modules_errored"] == 0
        assert report["issues"] == []
        assert report["modules_list"] == ["alpha_mod"]
        assert report["duplicate_registrations"] == []
        # Two of the 14 spot-check names are registered and succeed; the
        # other 12 hit the SKIP (not registered) branch.
        assert report["spot_checks_ok"] == 2
        assert report["spot_checks_err"] == 0
        assert report["logging_ok"] == 1
        assert report["logging_miss"] == 0
        out = capsys.readouterr().out
        assert "PASS - all tools real, documented, logged" in out
        assert "SKIP (not registered)" in out

    def test_mcp_init_failure_returns_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)

        def failing_initialize(**kwargs: Any) -> None:
            raise RuntimeError("sdk missing")

        monkeypatch.setattr("gnn.mcp.initialize", failing_initialize)

        rc = validate_tools.main()

        assert rc == 1
        out = capsys.readouterr().out
        assert "FATAL: Could not initialize MCP: sdk missing" in out

    def test_not_callable_and_undocumented_classified_as_issues(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "lambda_tool": _make_tool(func=lambda: {}, description="lambda tool"),
            "none_tool": _make_tool(func=None, description="no func"),
            "bare_tool": _make_tool(func=_real_func, description="   "),
            "ok_tool": _make_tool(func=_real_func, description="Real tool"),
        }
        _install_fake_mcp(monkeypatch, tools)

        rc = validate_tools.main()

        assert rc == 1
        issues = {
            (issue["tool"], issue["issue"])
            for issue in _read_report(src_root)["issues"]
        }
        assert ("lambda_tool", "not a real callable") in issues
        assert ("none_tool", "not a real callable") in issues
        assert ("bare_tool", "missing description") in issues

    def test_spot_check_failure_and_exception_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "list_analysis_tools": _make_tool(
                func=_real_func, description="returns success=False"
            ),
            "list_render_frameworks": _make_tool(func=_real_func, description="raises"),
        }

        def execute_tool(name: str, args: dict[str, Any]) -> Any:
            if name == "list_render_frameworks":
                raise RuntimeError("spot crash")
            return {"success": False, "error": "returned-false"}

        _install_fake_mcp(monkeypatch, tools, execute_tool_fn=execute_tool)

        rc = validate_tools.main()

        assert rc == 1
        report = _read_report(src_root)
        issues = {(issue["tool"], issue["issue"]) for issue in report["issues"]}
        assert ("list_analysis_tools", "execute returned success=False") in issues
        assert ("list_render_frameworks", "exception: spot crash") in issues
        assert report["spot_checks_ok"] == 0
        assert report["spot_checks_err"] == 2

    def test_logging_miss_issue_recorded(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        src_root = _seed_src_tree(
            tmp_path, good_modules=["alpha_mod"], bad_modules=["beta_mod"]
        )
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {"ok_tool": _make_tool(func=_real_func, description="Real tool")}
        _install_fake_mcp(monkeypatch, tools)

        rc = validate_tools.main()

        assert rc == 1
        report = _read_report(src_root)
        issues = {(issue["tool"], issue["issue"]) for issue in report["issues"]}
        assert ("beta_mod/mcp.py", "register_tools() has no logger.info") in issues
        assert report["logging_ok"] == 1
        assert report["logging_miss"] == 1


@pytest.mark.unit
class TestDuplicateRegistrations:
    """F5: duplicate register_tool names are detected at audit time."""

    def test_tracker_detects_duplicates_positional_and_kwarg(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import gnn.mcp.registry as registry_mod

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def dummy_register_tool(self: Any, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(
            registry_mod.MCPRegistryMixin, "register_tool", dummy_register_tool
        )
        mixin = registry_mod.MCPRegistryMixin.__new__(registry_mod.MCPRegistryMixin)
        tracker = validate_tools._DuplicateRegistrationTracker()
        tracker.install()
        try:
            mixin.register_tool("dup")
            mixin.register_tool(name="dup")
            mixin.register_tool("ok")
        finally:
            tracker.restore()

        assert tracker.duplicates == ["dup"]
        # The wrapped original saw every call, positional and kwarg forms.
        assert calls == [(("dup",), {}), ((), {"name": "dup"}), (("ok",), {})]
        assert registry_mod.MCPRegistryMixin.register_tool is dummy_register_tool

    def test_main_records_duplicate_registrations_in_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import gnn.mcp.registry as registry_mod

        src_root = _seed_src_tree(tmp_path, good_modules=[], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)

        def dummy_register_tool(self: Any, *args: Any, **kwargs: Any) -> None:
            return None

        monkeypatch.setattr(
            registry_mod.MCPRegistryMixin, "register_tool", dummy_register_tool
        )
        mixin = registry_mod.MCPRegistryMixin.__new__(registry_mod.MCPRegistryMixin)

        def registering_initialize(**kwargs: Any) -> None:
            mixin.register_tool("dup")
            mixin.register_tool(name="dup")

        _install_fake_mcp(
            monkeypatch,
            {"ok": _make_tool(func=_real_func, description="Real")},
        )
        monkeypatch.setattr("gnn.mcp.initialize", registering_initialize)

        rc = validate_tools.main()

        assert rc == 1
        report = _read_report(src_root)
        assert report["duplicate_registrations"] == ["dup"]
        issues = {(issue["tool"], issue["issue"]) for issue in report["issues"]}
        assert ("dup", "duplicate registration") in issues


class _GrowingTools:
    """len() grows for the first ``grow_reads`` reads, then stays fixed."""

    def __init__(self, final_len: int, grow_reads: int) -> None:
        self.reads = 0
        self._final = final_len
        self._grow_reads = grow_reads

    def __len__(self) -> int:
        self.reads += 1
        if self.reads >= self._grow_reads:
            return self._final
        return min(self._final, self.reads)


@pytest.mark.unit
class TestWaitForCensusStability:
    """Background registrations must settle before the census is captured."""

    def test_polls_until_counts_settle(self) -> None:
        m = SimpleNamespace(
            tools=_GrowingTools(final_len=7, grow_reads=2),
            modules={"only": 1},
        )

        validate_tools._wait_for_census_stability(
            m, poll_seconds=0.02, stable_seconds=0.06, deadline_seconds=5.0
        )

        assert len(m.tools) == 7
        # Kept polling past the growth instead of snapshotting early.
        assert m.tools.reads >= 4

    def test_returns_by_the_deadline(self) -> None:
        m = SimpleNamespace(
            tools=_GrowingTools(final_len=9, grow_reads=10_000),
            modules={},
        )

        validate_tools._wait_for_census_stability(
            m, poll_seconds=0.01, stable_seconds=0.5, deadline_seconds=0.05
        )

        assert len(m.tools) <= 9


@pytest.mark.unit
class TestWriteToolReference:
    """F4: the doc generator is a pure census-to-file function."""

    def _census(self) -> dict[str, Any]:
        return {
            "tools_list": [
                {
                    "name": "zeta_tool",
                    "module": "gnn.analysis",
                    "description": "Two\nlines",
                },
                {"name": "pipe_tool", "module": "meta", "description": "a|b | c"},
                {"name": "alpha_tool", "module": "analysis", "description": "First"},
            ],
            "modules_list": ["analysis", "meta"],
        }

    def test_generates_full_table_from_census(self, tmp_path: Path) -> None:
        doc_path = tmp_path / "docs" / "tool_reference.md"

        validate_tools.write_tool_reference(self._census(), doc_path)

        text = doc_path.read_text(encoding="utf-8")
        assert text.startswith("# GNN MCP Tool Quick Reference\n")
        assert "**3 tools across 2 modules**" in text
        assert "## Full Tool Table" in text
        # One row per tool, sorted by (domain, name); gnn. prefix stripped.
        assert "| analysis | `alpha_tool` | First |" in text
        assert "| analysis | `zeta_tool` | Two lines |" in text
        assert r"| meta | `pipe_tool` | a\|b \| c |" in text
        assert text.count("`alpha_tool`") == 1
        assert text.count("`zeta_tool`") == 1
        assert text.count("`pipe_tool`") == 1
        assert "subset" not in text
        for banned in ("legacy", "deprecated", "shim", "stub", "placeholder"):
            assert banned not in text
        assert (
            "Use `tests/mcp/test_mcp_audit.py` for the current registered "
            "tool/resource contract." in text
        )

    def test_main_markdown_mode_writes_doc_and_keeps_audit_rc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        src_root = _seed_src_tree(tmp_path, good_modules=["alpha_mod"], bad_modules=[])
        monkeypatch.setattr(validate_tools, "SRC_ROOT", src_root)
        tools = {
            "tool_a": _make_tool(func=_real_func, description="Does A"),
            "list_analysis_tools": _make_tool(
                func=_real_func, description="Lists analysis tools"
            ),
            "list_render_frameworks": _make_tool(
                func=_real_func, description="Lists render frameworks"
            ),
        }
        _install_fake_mcp(
            monkeypatch,
            tools,
            modules={"alpha_mod": SimpleNamespace(status="loaded", tools_count=3)},
            execute_tool_fn=lambda name, args: {"success": True},
        )
        doc_path = tmp_path / "out" / "tool_reference.md"

        rc = validate_tools.main(argv=["--markdown", str(doc_path)])

        assert rc == 0
        text = doc_path.read_text(encoding="utf-8")
        assert "**3 tools across 1 modules**" in text
        assert "| tests | `tool_a` | Does A |" in text
        assert _read_report(src_root)["tools_total"] == 3
        out = capsys.readouterr().out
        assert f"Tool reference saved → {doc_path}" in out
