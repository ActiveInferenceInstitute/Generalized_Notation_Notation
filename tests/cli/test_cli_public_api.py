"""Tests for cli module's public API surface not covered by existing tests.

Covers: get_module_info, FEATURES, __version__, _cmd_run, _cmd_parse,
_cmd_render (with missing file), _cmd_report, _cmd_preflight, _cmd_serve,
_cmd_lsp, _cmd_watch, _cmd_graph, _cmd_templates, _cmd_pull, and
``gnn.cli.commands.find_render_artifact``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestCLIConstants:
    """Test module-level constants."""

    def test_features_dict(self) -> None:
        import gnn.cli as cli

        assert hasattr(cli, "FEATURES")
        assert isinstance(cli.FEATURES, dict)
        for key in (
            "subcommands",
            "pipeline_execution",
            "file_validation",
            "file_parsing",
            "render_dispatch",
            "lsp_launch",
        ):
            assert key in cli.FEATURES

    def test_version(self) -> None:
        import gnn.cli as cli

        assert hasattr(cli, "__version__")
        assert isinstance(cli.__version__, str)

    def test_get_module_info(self) -> None:
        from gnn.cli import get_module_info

        info = get_module_info()
        assert isinstance(info, dict)
        assert "version" in info
        assert "description" in info


class TestCmdHandlers:
    """Test individual command handlers."""

    def test_cmd_parse_missing_file(self) -> None:
        from gnn.cli import _cmd_parse

        args = argparse.Namespace(file=Path("/nonexistent.md"), format="json")
        result = _cmd_parse(args)
        assert result == 1  # Error exit code

    def test_cmd_parse_keeps_interface_separate_from_pomdp(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        from gnn.cli import _cmd_parse

        model = tmp_path / "interface.gnn.md"
        model.write_text(
            "## GNNSection\nGNN\n## ModelName\nWeather interface\n"
            "## StateSpaceBlock\nexternal_temperature[1,type=float]\n"
            "observed_temperature[1,type=float]\n## Connections\n"
            "external_temperature>observed_temperature\n"
            "## InitialParameterization\nthreshold = {2}\n",
            encoding="utf-8",
        )
        assert _cmd_parse(argparse.Namespace(file=model, format="json", json=True)) == 0
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "success"
        payload = envelope["data"]
        assert len(payload["variables"]) == 2
        assert len(payload["connections"]) == 1
        assert payload["errors"] == []
        assert "pomdp" not in payload
        assert "source-declared" in payload["warnings"][-1]

    def test_cmd_parse_preserves_source_declared_pomdp(self, capsys: Any) -> None:
        from gnn.cli import _cmd_parse

        model = (
            Path(__file__).resolve().parents[2]
            / "input/gnn_files/discrete/two_state_bistable.md"
        )
        assert _cmd_parse(argparse.Namespace(file=model, format="json", json=True)) == 0
        payload = json.loads(capsys.readouterr().out)["data"]
        assert (
            payload["pomdp"]["matrix_provenance"]["A"]["source"]
            == "InitialParameterization"
        )
        assert payload["pomdp"]["num_states"] == 2

    def test_cmd_validate_warning_uses_exit_code_two(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Non-strict validation findings are warnings, not successes."""
        from gnn.cli import _cmd_validate

        malformed = tmp_path / "malformed.md"
        malformed.write_text("# incomplete GNN\n", encoding="utf-8")
        result = _cmd_validate(
            argparse.Namespace(file=malformed, strict=False, json=True)
        )
        assert result == 2
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "warning"

    def test_cmd_parse_reports_parser_warnings_with_exit_code_two(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Partial parse diagnostics are preserved in JSON output and status."""
        from gnn.cli import _cmd_parse

        malformed = tmp_path / "malformed.md"
        malformed.write_text(
            "## StateSpaceBlock\nA[not-a-dimension,type=float]\n"
            "## Connections\nA>missing\n",
            encoding="utf-8",
        )
        result = _cmd_parse(
            argparse.Namespace(file=malformed, format="json", json=True)
        )
        assert result == 2
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["status"] == "warning"
        assert envelope["data"]["warnings"]

    def test_cmd_render_missing_file(self) -> None:
        from gnn.cli import _cmd_render

        args = argparse.Namespace(
            file=Path("/nonexistent.md"),
            framework="pymdp",
            output=None,
            verbose=False,
        )
        result = _cmd_render(args)
        assert result == 1

    def test_cmd_report_missing_dir(self) -> None:
        from gnn.cli import _cmd_report

        args = argparse.Namespace(output_dir=Path("/nonexistent_output"))
        result = _cmd_report(args)
        assert result == 1

    def test_cmd_report_json_missing_dir(self, capsys: Any) -> None:
        from gnn.cli import _cmd_report

        args = argparse.Namespace(output_dir=Path("/nonexistent_output"), json=True)
        result = _cmd_report(args)
        assert result == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "error"
        assert "not found" in envelope["error"]

    def test_cmd_graph_json_missing_file(self, capsys: Any) -> None:
        from gnn.cli import _cmd_graph

        args = argparse.Namespace(
            file=Path("/nonexistent.md"), format="mermaid", json=True
        )
        result = _cmd_graph(args)
        assert result == 1
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "error"
        assert "not found" in envelope["error"]

    def test_cmd_health_json(self, capsys: Any) -> None:
        from gnn.cli import _cmd_health

        args = argparse.Namespace(strict=False, json=True)
        result = _cmd_health(args)
        assert isinstance(result, int)
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] in ("success", "warning", "error")
        assert "renderers" in envelope["data"]
        assert "environment" in envelope["data"]

    def test_cmd_preflight_default(self) -> None:
        from gnn.cli import _cmd_preflight

        args = argparse.Namespace(config=None, verbose=False, json=False)
        result = _cmd_preflight(args)
        assert isinstance(result, int)

    def test_cmd_preflight_json(self, capsys: Any) -> None:
        from gnn.cli import _cmd_preflight

        args = argparse.Namespace(config=None, verbose=False, json=True)
        result = _cmd_preflight(args)
        assert isinstance(result, int)
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] in ("success", "warning", "error")
        assert "checks_passed" in envelope["data"]

    def test_cmd_serve_default(self) -> None:
        # _cmd_serve starts a blocking uvicorn server (never returns), so we
        # verify the handler wiring instead: api.app must be importable and
        # the serve handler must accept the documented args.
        import gnn.cli as cli

        assert hasattr(cli, "_cmd_serve")
        import gnn.api.app  # noqa: F401

    def test_cmd_lsp(self) -> None:
        # _cmd_lsp calls start_server() which tries to start stdio server.
        # Instead, test that the handler function exists and is callable.
        from gnn.cli import _cmd_lsp

        assert callable(_cmd_lsp)

    def test_cmd_graph_missing_file(self) -> None:
        from gnn.cli import _cmd_graph

        args = argparse.Namespace(
            file=Path("/nonexistent.md"), format="mermaid", verbose=False
        )
        result = _cmd_graph(args)
        assert result == 1

    def test_cmd_models_json(self, capsys: Any) -> None:
        from gnn.cli import _cmd_models

        args = argparse.Namespace(
            target_dir=Path("input/gnn_files/basics"),
            query_ontology=None,
            json=True,
        )
        result = _cmd_models(args)
        assert result == 0
        captured = capsys.readouterr()
        envelope = json.loads(captured.out)
        assert envelope["status"] == "success"
        assert "matching_models" in envelope["data"]
        assert len(envelope["data"]["matching_models"]) >= 1

    def test_cmd_pull_missing_template(self, tmp_path: Path) -> None:
        from gnn.cli import _cmd_pull

        args = argparse.Namespace(
            name="nonexistent-template",
            output_dir=tmp_path,
            dry_run=True,
            overwrite=False,
            verbose=False,
        )
        result = _cmd_pull(args)
        assert result == 1

    def test_cmd_templates_unknown_subcommand(self, capsys: Any) -> None:
        from gnn.cli import _cmd_templates

        args = argparse.Namespace(
            templates_command="nonexistent",
            verbose=False,
        )
        result = _cmd_templates(args)
        assert isinstance(result, int)

    def test_cmd_watch_handler_exists(self) -> None:
        # _cmd_watch starts a blocking filesystem watcher (GNNWatcher.start()
        # loops until stopped), so we verify the handler is present and
        # callable rather than invoking it in-process.
        from gnn.cli import _cmd_watch

        assert callable(_cmd_watch)


class TestFindRenderArtifact:
    """Test find_render_artifact helper."""

    def test_empty_dir_returns_none(self, tmp_path: Path) -> None:
        from gnn.cli.commands import find_render_artifact

        result = find_render_artifact(tmp_path, "pymdp")
        assert result is None

    def test_finds_py_file(self, tmp_path: Path) -> None:
        from gnn.cli.commands import find_render_artifact

        output = tmp_path / "pymdp" / "model.py"
        output.parent.mkdir(parents=True)
        output.write_text("code")
        result = find_render_artifact(tmp_path, "pymdp")
        assert result == output

    def test_finds_toml_for_rxinfer(self, tmp_path: Path) -> None:
        from gnn.cli.commands import find_render_artifact

        output = tmp_path / "rxinfer" / "model.toml"
        output.parent.mkdir(parents=True)
        output.write_text("config")
        result = find_render_artifact(tmp_path, "rxinfer")
        assert result == output

    def test_skips_known_non_artifact_files(self, tmp_path: Path) -> None:
        from gnn.cli.commands import find_render_artifact

        (tmp_path / "README.md").write_text("readme")
        (tmp_path / "processing_summary.json").write_text("{}")
        result = find_render_artifact(tmp_path, "pymdp")
        assert result is None

    def test_finds_from_summary_json(self, tmp_path: Path) -> None:
        import hashlib

        from gnn.cli.commands import find_render_artifact as legacy_find_artifact
        from gnn.render import find_render_artifact

        summary = tmp_path / "render_processing_summary.json"
        output_file = tmp_path / "pymdp" / "model.py"
        output_file.parent.mkdir()
        output_file.write_text("code")
        summary.write_text(
            json.dumps(
                {
                    "file_results": {
                        "model.md": {
                            "framework_results": {
                                "pymdp": {
                                    "success": True,
                                    "output_files": [str(output_file)],
                                    "artifact_identities": [
                                        {
                                            "path": str(output_file.resolve()),
                                            "sha256": hashlib.sha256(
                                                output_file.read_bytes()
                                            ).hexdigest(),
                                        }
                                    ],
                                }
                            }
                        }
                    }
                }
            )
        )
        for find_artifact in (find_render_artifact, legacy_find_artifact):
            assert find_artifact(tmp_path, "pymdp", require_current=True) == output_file
        output_file.write_text("changed after receipt")
        for find_artifact in (find_render_artifact, legacy_find_artifact):
            assert find_artifact(tmp_path, "pymdp", require_current=True) is None
        assert legacy_find_artifact(tmp_path, "pymdp") == output_file
