"""Composability and behavior-pin tests for the refactored ``cli`` module.

Covers the surface added/changed in the 2026-09-04 fleet pass:
- ``build_parser`` as a pure parser factory (introspectable, no dispatch)
- ``COMMAND_HANDLERS``/``SUBCOMMANDS`` as the single dispatch source of truth
- ``CommandHandler`` typed handler contract
- envelope ``meta.command`` field on JSON output
- ``gnn parse --format yaml`` real YAML output (with JSON degradation note)
- ``_guard_input_file`` shared missing-file guard
- ``cli.lsp`` injectable-transport layering (diagnose_text, run_lsp_loop)
- exit-code normalization (no bare ``return 1``/``return 0`` escapes)
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gnn.cli as cli
from gnn.cli import lsp as cli_lsp

REPO = Path(__file__).resolve().parents[2]
ACTINF_EXEMPLAR = REPO / "input" / "gnn_files" / "discrete" / "actinf_pomdp_agent.md"
SIMPLE_MDP_EXEMPLAR = REPO / "input" / "gnn_files" / "discrete" / "simple_mdp.md"


def _subparser_map(
    parser: argparse.ArgumentParser,
) -> dict[str, argparse.ArgumentParser]:
    """Narrow argparse's private subparser registry to its choices mapping."""
    subparsers = parser._subparsers
    assert subparsers is not None
    group_choices = subparsers._group_actions[0].choices
    assert isinstance(group_choices, dict)
    return group_choices


# ---------------------------------------------------------------------------
# build_parser — pure factory, introspectable surface
# ---------------------------------------------------------------------------


class TestBuildParser:
    """The parser factory exposes the full subcommand surface."""

    def test_build_parser_returns_fresh_parser_each_call(self) -> None:
        p1 = cli.build_parser()
        p2 = cli.build_parser()
        assert p1 is not p2

    def test_subcommand_names_match_dispatch_table(self) -> None:
        parser = cli.build_parser()
        choices = set(_subparser_map(parser))
        assert choices == set(cli.COMMAND_HANDLERS)
        assert choices == set(cli.SUBCOMMANDS)

    def test_subcommands_sorted_and_complete(self) -> None:
        assert cli.SUBCOMMANDS == tuple(sorted(cli.SUBCOMMANDS))
        assert len(cli.SUBCOMMANDS) == 20

    def test_extract_flags_introspectable(self) -> None:
        parser = cli.build_parser()
        extract = _subparser_map(parser)["extract"]
        option_strings = {o for a in extract._actions for o in a.option_strings}
        assert {"--strict", "--no-strict", "--compact"} <= option_strings

    def test_parse_format_choices(self) -> None:
        parser = cli.build_parser()
        parse_p = _subparser_map(parser)["parse"]
        fmt = next(a for a in parse_p._actions if a.dest == "format")
        assert fmt.choices == ["json", "yaml", "summary"]


# ---------------------------------------------------------------------------
# Dispatch table — single source of truth
# ---------------------------------------------------------------------------


class TestDispatchTable:
    """COMMAND_HANDLERS names resolve to real module callables."""

    @pytest.mark.parametrize(
        "command",
        [
            "run",
            "validate",
            "parse",
            "extract",
            "render",
            "report",
            "reproduce",
            "preflight",
            "health",
            "serve",
            "templates",
            "models",
            "pull",
            "lsp",
            "watch",
            "graph",
            "gui",
            "mcp",
        ],
    )
    def test_handler_attr_exists_and_is_callable(self, command: str) -> None:
        handler_name = cli.COMMAND_HANDLERS[command]
        handler = getattr(cli, handler_name)
        assert callable(handler)

    def test_table_covers_all_documented_subcommands(self) -> None:
        expected = {
            "run",
            "validate",
            "parse",
            "extract",
            "render",
            "report",
            "reproduce",
            "preflight",
            "health",
            "serve",
            "templates",
            "models",
            "pull",
            "watch",
            "mcp",
            "graph",
            "gui",
            "lsp",
            "complexity",
            "benchmark",
        }
        assert set(cli.COMMAND_HANDLERS) == expected


# ---------------------------------------------------------------------------
# Envelope — meta.command propagation
# ---------------------------------------------------------------------------


class TestEnvelopeMeta:
    """Envelope meta always carries version; command merges additively."""

    def test_meta_defaults_to_version_only(self) -> None:
        env = cli._envelope("success")
        assert env["meta"] == {"version": cli.__version__}

    def test_meta_command_added(self) -> None:
        env = cli._envelope("error", error="boom", command="validate")
        assert env["meta"]["command"] == "validate"
        assert env["meta"]["version"] == cli.__version__

    def test_explicit_meta_wins_over_command(self) -> None:
        env = cli._envelope("success", meta={"version": "9.9.9"}, command="parse")
        assert env["meta"]["version"] == "9.9.9"
        assert env["meta"]["command"] == "parse"

    def test_validate_json_envelope_includes_command(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        template = (
            REPO / "src" / "gnn" / "cli" / "template_assets" / "actinf_pomdp_2state.md"
        )
        assert cli.main(["validate", str(template), "--json"]) == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["meta"]["command"] == "validate"
        assert envelope["meta"]["version"] == cli.__version__


# ---------------------------------------------------------------------------
# parse --format yaml — real YAML output with JSON degradation
# ---------------------------------------------------------------------------


class TestParseYaml:
    """`gnn parse --format yaml` emits parseable YAML, not JSON."""

    def test_yaml_output_is_parseable_yaml(self, capsys: Any) -> None:
        import yaml

        code = cli.main(["parse", str(ACTINF_EXEMPLAR), "--format", "yaml"])
        assert code == 0
        out = capsys.readouterr().out
        payload = yaml.safe_load(out)
        assert isinstance(payload, dict)
        assert payload["file"] == str(ACTINF_EXEMPLAR)
        assert isinstance(payload["variables"], list)
        # YAML output must not be JSON (json would still parse as YAML,
        # so pin the distinction through the document start absence).
        assert not out.lstrip().startswith("{")

    def test_yaml_degrades_to_json_without_pyyaml(
        self, capsys: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("no yaml in simulated environment")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        code = cli.main(["parse", str(ACTINF_EXEMPLAR), "--format", "yaml"])
        assert code == 0
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["file"] == str(ACTINF_EXEMPLAR)

    def test_render_yaml_pure_function(self) -> None:
        text = cli._render_yaml({"a": 1})
        assert text is not None
        assert "a: 1" in text


# ---------------------------------------------------------------------------
# _guard_input_file — shared missing-file guard
# ---------------------------------------------------------------------------


class TestGuardInputFile:
    """The shared guard logs, optionally emits envelope, returns bool."""

    def test_missing_file_emits_envelope_when_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ok = cli._guard_input_file(
            Path("/nonexistent.md"), json_output=True, command="parse"
        )
        assert ok is False
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["meta"]["command"] == "parse"
        assert "not found" in envelope["error"]

    def test_missing_file_silent_when_not_json(self, capsys: Any) -> None:
        ok = cli._guard_input_file(
            Path("/nonexistent.md"), json_output=False, command="graph"
        )
        assert ok is False
        assert capsys.readouterr().out == ""

    def test_existing_file_passes(self, tmp_path: Path) -> None:
        f = tmp_path / "model.md"
        f.write_text("x", encoding="utf-8")
        assert cli._guard_input_file(f, json_output=False, command="parse") is True


# ---------------------------------------------------------------------------
# cli.lsp — pure diagnostics + injectable transport loop
# ---------------------------------------------------------------------------

# Minimal fully-valid GNN document: all parse-level required sections, one
# declared/connected variable pair, no parameterization to cross-check.
_VALID_MINIMAL_GNN = (
    "## GNNSection\n"
    "ActInfPOMDP\n"
    "\n"
    "## GNNVersionAndFlags\n"
    "GNN v1\n"
    "\n"
    "## ModelName\n"
    "Minimal Model\n"
    "\n"
    "## StateSpaceBlock\n"
    "s_f[2,1,type=float]\n"
    "s_x[2,1,type=float]\n"
    "\n"
    "## Connections\n"
    "s_f>s_x\n"
    "\n"
    "## Time\n"
    "Dynamic\n"
    "\n"
    "## Footer\n"
    "End.\n"
)


class TestLspDiagnoseText:
    """diagnose_text is pure: same input, same diagnostics, no I/O."""

    def test_unclosed_brace_flagged(self) -> None:
        diags = cli_lsp.diagnose_text("{ unclosed")
        assert diags[0]["severity"] == 1
        assert diags[0]["message"] == "Missing closing brace '}'"
        # Schema validation runs alongside the brace check: a brace-only
        # fragment is also missing every required GNN section.
        assert any("Missing required section" in d["message"] for d in diags[1:])

    def test_clean_text_has_no_diagnostics(self) -> None:
        assert cli_lsp.diagnose_text(_VALID_MINIMAL_GNN) == []

    def test_no_text_no_diagnostics(self) -> None:
        assert cli_lsp.diagnose_text("") == []


class TestLspLoopInjection:
    """run_lsp_loop drives a full session over injected callables."""

    def _session(self, messages: list[dict[str, Any]]) -> tuple[list[Any], int]:
        written: list[Any] = []
        inbox = iter(messages)

        def reader() -> Any:
            try:
                return next(inbox)
            except StopIteration:
                return None

        def writer(msg: Any, *args: Any) -> None:
            written.append(msg)

        # Responses use the injected writer; diagnostics currently resolve
        # the module-level sink. Capture that boundary too, without changing
        # production transport behavior to satisfy this test.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(cli_lsp, "write_message", writer)
            cli_lsp.run_lsp_loop(reader, writer)
        return written, len(written)

    def test_initialize_and_shutdown_roundtrip(self) -> None:
        written, _ = self._session(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
                {"jsonrpc": "2.0", "method": "exit"},
            ]
        )
        assert written[0]["result"]["serverInfo"]["name"] == "gnn-lsp"
        assert written[1] == {"jsonrpc": "2.0", "id": 2, "result": None}

    def test_unknown_request_gets_method_not_found(self) -> None:
        written, _ = self._session([{"jsonrpc": "2.0", "id": 7, "method": "no/such"}])
        assert written[0]["error"]["code"] == -32601

    def test_unknown_notification_is_silent(self) -> None:
        written, count = self._session([{"jsonrpc": "2.0", "method": "no/such"}])
        assert count == 0
        assert written == []

    def test_did_open_publishes_diagnostics(self) -> None:
        written, _ = self._session(
            [
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": {
                        "textDocument": {"uri": "file:///m.md", "text": "{ open"}
                    },
                },
                {"jsonrpc": "2.0", "method": "exit"},
            ]
        )
        assert written[0]["method"] == "textDocument/publishDiagnostics"
        assert written[0]["params"]["diagnostics"][0]["severity"] == 1


class TestLspStreamDefaults:
    """read/write fall back to live sys.stdin/stdout at call time."""

    def test_write_message_targets_swapped_stdout(
        self, capsys: pytest.CaptureFixture[str], monkeypatch: Any
    ) -> None:
        buf = io.StringIO()
        monkeypatch.setattr(sys, "stdout", buf)
        cli_lsp.write_message({"jsonrpc": "2.0", "id": 1})
        assert "Content-Length:" in buf.getvalue()

    def test_read_message_from_injected_stream(self) -> None:
        body = '{"jsonrpc": "2.0", "id": 7}'
        stream = io.StringIO(f"Content-Length: {len(body)}\r\n\r\n{body}")
        msg = cli_lsp.read_message(stream)
        assert msg["id"] == 7


# ---------------------------------------------------------------------------
# Handler signature conformance
# ---------------------------------------------------------------------------


class TestHandlerSignatures:
    """Every dispatch-table handler accepts Namespace and returns int."""

    @pytest.mark.parametrize("command", sorted(cli.COMMAND_HANDLERS))
    def test_handler_returns_int_for_minimal_args(
        self, command: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = getattr(cli, cli.COMMAND_HANDLERS[command])
        # A missing file guards file commands only. Run/serve/watch/LSP have
        # live execution boundaries and must never launch from this unit test.
        # Patch the symbol each handler imports, keeping real CLI dispatch.
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def completed(*args: Any, **kwargs: Any) -> int:
            calls.append((args, kwargs))
            return cli.EXIT_WARNING if command == "run" else cli.EXIT_SUCCESS

        if command == "run":
            import gnn.main as pipeline_module

            monkeypatch.setattr(pipeline_module, "main", completed)
        elif command == "serve":
            from gnn.api import app as api_app

            monkeypatch.setattr(api_app, "start_server", completed)
        elif command == "lsp":
            monkeypatch.setattr(cli_lsp, "start_lsp", completed)
        elif command == "watch":
            from gnn.cli.watcher import GNNWatcher

            monkeypatch.setattr(GNNWatcher, "start", completed)
        elif command == "gui":
            import gnn.gui as gui_module

            monkeypatch.setattr(gui_module, "process_gui", completed)
        elif command == "mcp":
            import gnn.mcp as mcp_module

            monkeypatch.setattr(mcp_module, "initialize", lambda: None)
            monkeypatch.setattr(
                mcp_module,
                "get_mcp_instance",
                lambda: _StubMcpInstance([]),
            )
        elif command in {"health", "preflight"}:
            from gnn.pipeline import preflight
            from gnn.render import health

            monkeypatch.setattr(
                preflight, "check_environment", preflight.PreflightReport
            )
            monkeypatch.setattr(
                preflight, "run_preflight", lambda **kwargs: preflight.PreflightReport()
            )
            monkeypatch.setattr(health, "check_renderers", dict)

        ns = argparse_namespace_for(command, missing_file=True)
        original_argv = sys.argv
        result = handler(ns)
        if command in {"run", "serve", "lsp", "watch", "gui"}:
            assert len(calls) == 1
        if command == "run":
            assert result == cli.EXIT_WARNING
            assert sys.argv is original_argv
        if command == "serve":
            assert calls == [((), {"host": ns.host, "port": ns.port})]
        if command == "watch":
            assert calls[0][0][0].watch_dir == ns.dir
        assert isinstance(result, int)
        assert result in (cli.EXIT_SUCCESS, cli.EXIT_ERROR, cli.EXIT_WARNING)


def argparse_namespace_for(command: str, *, missing_file: bool) -> argparse.Namespace:
    """Build a minimal Namespace covering each handler's touched attrs."""
    missing = Path("/nonexistent/for-sure.md")
    file_arg = missing if missing_file else Path("x")
    common: dict[str, Any] = {"verbose": False}
    per_command: dict[str, dict[str, Any]] = {
        "run": {
            "target_dir": "in",
            "output_dir": "out",
            "log_format": "human",
            "skip_llm": False,
            "skip_steps": None,
            "only_steps": None,
        },
        "validate": {"file": file_arg, "strict": False, "json": False},
        "parse": {"file": file_arg, "format": "json", "json": False},
        "extract": {"file": file_arg, "strict": True, "compact": False, "json": False},
        "render": {
            "file": file_arg,
            "framework": "pymdp",
            "output": None,
            "json": False,
        },
        "report": {"output_dir": Path("/nonexistent_output"), "json": False},
        "reproduce": {"run_hash": "abc123def456", "history_dir": Path("no/such")},
        "preflight": {"config": None, "json": False},
        "health": {"strict": False, "json": False},
        "serve": {"host": "127.0.0.1", "port": 8000, "surface": "runs"},
        "templates": {"templates_command": None, "json": False},
        "models": {
            "target_dir": Path("input/gnn_files"),
            "query_ontology": None,
            "json": False,
        },
        "pull": {
            "name": "nonexistent-template",
            "output_dir": Path("in"),
            "dry_run": True,
            "overwrite": False,
            "json": False,
        },
        "lsp": {},
        "watch": {"dir": file_arg},
        "graph": {"file": file_arg, "format": "mermaid", "json": False},
        "gui": {
            "target_dir": "in",
            "output_dir": "out",
            "gui_types": "gui_1,gui_2",
            "interactive": False,
            "open_browser": False,
            "launch_editor": False,
        },
        "mcp": {"mcp_command": None, "json": False, "name": None},
        "complexity": {"path": file_arg, "json": False},
        "benchmark": {
            "target_dir": Path("/nonexistent/for-sure-dir"),
            "output_dir": Path("out"),
            "frameworks": "all",
            "repeats": 1,
            "json": False,
        },
    }
    return argparse.Namespace(**common, **per_command[command])


class _StubMcpInstance:
    """Minimal mirror of the real MCP instance's registry accessors."""

    def __init__(self, tools: list[dict[str, Any]]) -> None:
        self._tools = tools

    def list_available_tools(
        self, include_metadata: bool = True
    ) -> list[dict[str, Any]]:
        return list(self._tools)

    def get_tool_info(self, tool_name: str) -> dict[str, Any] | None:
        for tool in self._tools:
            if tool["name"] == tool_name:
                return dict(tool)
        return None


# ---------------------------------------------------------------------------
# serve --surface — API surface selection and dispatch routing
# ---------------------------------------------------------------------------


class TestServeSurface:
    """``serve`` routes the runs, jobs, and both API surfaces, plus website."""

    def test_surface_choices_and_default(self) -> None:
        parser = cli.build_parser()
        serve_p = _subparser_map(parser)["serve"]
        surface = next(a for a in serve_p._actions if a.dest == "surface")
        assert surface.choices == ["runs", "jobs", "both", "website"]
        assert surface.default == "runs"

    def test_runs_surface_calls_start_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.api import app as api_app

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(api_app, "start_server", record)
        args = argparse.Namespace(
            surface="runs", host="127.0.0.1", port=8000, verbose=False
        )
        assert cli._cmd_serve(args) == cli.EXIT_SUCCESS
        assert calls == [((), {"host": "127.0.0.1", "port": 8000})]

    def test_jobs_surface_calls_run_server(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.api import server as api_server

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(api_server, "run_server", record)
        args = argparse.Namespace(
            surface="jobs", host="127.0.0.1", port=8100, verbose=False
        )
        assert cli._cmd_serve(args) == cli.EXIT_SUCCESS
        assert calls == [((), {"host": "127.0.0.1", "port": 8100})]

    def test_both_surface_starts_jobs_on_port_plus_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import uvicorn

        from gnn.api import app as api_app
        from gnn.api import server as api_server

        start_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        job_configs: list[Any] = []

        def record_start(*args: Any, **kwargs: Any) -> None:
            start_calls.append((args, kwargs))

        class _StubJobsServer:
            def __init__(self, config: Any) -> None:
                job_configs.append(config)

            def run(self) -> None:
                return None

        monkeypatch.setattr(api_app, "start_server", record_start)
        monkeypatch.setattr(api_server, "create_app", lambda: object())
        monkeypatch.setattr(uvicorn, "Server", _StubJobsServer)
        args = argparse.Namespace(
            surface="both", host="127.0.0.1", port=8200, verbose=False
        )
        assert cli._cmd_serve(args) == cli.EXIT_SUCCESS
        assert start_calls == [((), {"host": "127.0.0.1", "port": 8200})]
        assert len(job_configs) == 1
        assert job_configs[0].port == 8201
        assert job_configs[0].host == "127.0.0.1"


# ---------------------------------------------------------------------------
# extract/render --json — standard CLI envelope output
# ---------------------------------------------------------------------------


class TestJsonEnvelopeFlags:
    """``extract`` and ``render`` gain the standard --json envelope mode."""

    def test_extract_and_render_json_flags_introspectable(self) -> None:
        parser = cli.build_parser()
        choices = _subparser_map(parser)
        for name in ("extract", "render"):
            option_strings = {
                o for a in choices[name]._actions for o in a.option_strings
            }
            assert "--json" in option_strings

    def test_extract_json_envelope_on_exemplar(self, capsys: Any) -> None:
        assert SIMPLE_MDP_EXEMPLAR.is_file()
        assert cli.main(["extract", str(SIMPLE_MDP_EXEMPLAR), "--json"]) == (
            cli.EXIT_SUCCESS
        )
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "success"
        assert envelope["meta"]["command"] == "extract"
        assert isinstance(envelope["data"], dict)

    def test_render_json_envelope_with_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        import gnn.render as render_module

        source = tmp_path / "model.md"
        source.write_text("# GNN stub model\n", encoding="utf-8")
        output = tmp_path / "rendered" / "model.py"

        def fake_process_render(
            *, target_dir: Path, output_dir: Path, **kwargs: Any
        ) -> bool:
            framework_dir = output_dir / "pymdp"
            framework_dir.mkdir(parents=True, exist_ok=True)
            (framework_dir / "model.py").write_text(
                "# stub artifact\n", encoding="utf-8"
            )
            import hashlib

            artifact = (framework_dir / "model.py").resolve()
            (output_dir / "render_processing_summary.json").write_text(
                json.dumps(
                    {
                        "file_results": {
                            "model": {
                                "framework_results": {
                                    "pymdp": {
                                        "success": True,
                                        "output_files": [str(artifact)],
                                        "artifact_identities": [
                                            {
                                                "path": str(artifact),
                                                "sha256": hashlib.sha256(
                                                    artifact.read_bytes()
                                                ).hexdigest(),
                                            }
                                        ],
                                    },
                                }
                            }
                        }
                    }
                )
            )
            return True

        monkeypatch.setattr(render_module, "process_render", fake_process_render)
        assert (
            cli.main(
                [
                    "render",
                    str(source),
                    "--framework",
                    "pymdp",
                    "--output",
                    str(output),
                    "--json",
                ]
            )
            == cli.EXIT_SUCCESS
        )
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "success"
        assert envelope["meta"]["command"] == "render"
        assert envelope["data"]["file"] == str(source)
        assert envelope["data"]["framework"] == "pymdp"
        assert envelope["data"]["output"] == str(output)
        assert envelope["data"]["render_dir"]
        assert envelope["data"]["artifact"] is not None

    def test_render_json_error_envelope_on_missing_file(self, capsys: Any) -> None:
        assert (
            cli.main(["render", "/nonexistent/for-sure.md", "--json"]) == cli.EXIT_ERROR
        )
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["meta"]["command"] == "render"
        assert envelope["error"]["code"] == "render_error"


# ---------------------------------------------------------------------------
# gnn mcp — tool-surface inspection over the lazy gnn.mcp bridge
# ---------------------------------------------------------------------------


class TestMcpSubcommand:
    """``gnn mcp list|info`` wraps gnn.mcp registry access in the envelope."""

    TOOLS: list[dict[str, Any]] = [
        {
            "name": "b.tool",
            "module": "m_two",
            "category": "cat_two",
            "description": "second tool",
            "version": "1.0.0",
        },
        {
            "name": "a.tool",
            "module": "m_one",
            "category": "cat_one",
            "description": "first tool",
            "version": "1.0.0",
        },
    ]

    def _install_stub(self, monkeypatch: pytest.MonkeyPatch) -> _StubMcpInstance:
        import gnn.mcp as mcp_module

        stub = _StubMcpInstance([dict(tool) for tool in self.TOOLS])
        monkeypatch.setattr(mcp_module, "initialize", lambda: None)
        monkeypatch.setattr(mcp_module, "get_mcp_instance", lambda: stub)
        return stub

    def test_mcp_list_json_envelope(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        self._install_stub(monkeypatch)
        assert cli.main(["mcp", "list", "--json"]) == cli.EXIT_SUCCESS
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "success"
        assert envelope["meta"]["command"] == "mcp"
        assert envelope["data"]["total"] == 2
        assert [t["name"] for t in envelope["data"]["tools"]] == ["a.tool", "b.tool"]
        assert envelope["data"]["tools"][0]["module"] == "m_one"
        assert envelope["data"]["tools"][0]["category"] == "cat_one"

    def test_mcp_info_json_envelope(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        self._install_stub(monkeypatch)
        assert cli.main(["mcp", "info", "a.tool", "--json"]) == cli.EXIT_SUCCESS
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "success"
        assert envelope["meta"]["command"] == "mcp"
        assert envelope["data"]["name"] == "a.tool"
        assert envelope["data"]["module"] == "m_one"
        assert envelope["data"]["category"] == "cat_one"

    def test_mcp_info_unknown_tool_json_error_envelope(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        self._install_stub(monkeypatch)
        assert cli.main(["mcp", "info", "nope.tool", "--json"]) == cli.EXIT_ERROR
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "error"
        assert envelope["meta"]["command"] == "mcp"
        assert envelope["error"]["code"] == "unknown_tool"

    def test_mcp_info_unknown_tool_human_mode_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, capsys: Any
    ) -> None:
        self._install_stub(monkeypatch)
        assert cli.main(["mcp", "info", "nope.tool"]) == cli.EXIT_ERROR
        assert capsys.readouterr().out == ""
