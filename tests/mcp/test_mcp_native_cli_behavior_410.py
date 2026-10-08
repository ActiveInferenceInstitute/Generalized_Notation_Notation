"""Installed MCP CLI consumers expose actual registered tools and diagnostics."""

import json
import logging
import re
import sys

import pytest

from gnn import get_module_info
from gnn.mcp import cli, initialize

pytestmark = pytest.mark.mcp


@pytest.fixture
def native_cli(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Provision real GNN/meta tools and keep CLI process state local to a test."""
    registry, sdk_found, loaded = initialize(
        modules_allowlist=["gnn"], force_refresh=True
    )
    assert sdk_found and loaded
    assert "get_gnn_module_info" in registry.tools
    monkeypatch.chdir(tmp_path)
    root = logging.getLogger()
    original_logger = cli.logger
    loggers = (root, original_logger, logging.getLogger("mcp.cli"))
    states = [(logger, logger.level, list(logger.handlers)) for logger in loggers]
    levels = {
        handler: handler.level for _, _, handlers in states for handler in handlers
    }

    def invoke(*arguments):
        monkeypatch.setattr(sys, "argv", ["gnn-mcp", *arguments])
        cli.main()

    yield invoke, registry

    for logger, level, handlers in states:
        for handler in logger.handlers:
            if handler not in handlers:
                handler.close()
        logger.handlers[:] = handlers
        logger.setLevel(level)
    for handler, level in levels.items():
        handler.setLevel(level)
    cli.logger = original_logger


def test_native_cli_list_resolves_actual_tool_and_resource_metadata(
    native_cli, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke, registry = native_cli
    capsys.readouterr()

    invoke("--format", "json", "--verbose", "list")

    document = json.loads(capsys.readouterr().out)
    tools = {tool["name"]: tool for tool in document["tools"]}
    assert (
        tools["get_gnn_documentation"]["schema"]
        == registry.tools["get_gnn_documentation"].schema
    )
    assert (
        tools["get_gnn_module_info"]["module"]
        == registry.tools["get_gnn_module_info"].module
    )
    assert any(
        resource["uri_template"] == "gnn://documentation/{doc_name}"
        for resource in document["resources"]
    )


def test_native_cli_executes_bundled_document_resource_and_reports_real_usage(
    native_cli, capsys: pytest.CaptureFixture[str]
) -> None:
    invoke, _ = native_cli
    capsys.readouterr()
    invoke("--format", "json", "execute", "get_gnn_module_info")
    module = json.loads(capsys.readouterr().out)
    assert module["description"] == get_module_info()["description"]
    assert "parse_gnn_content" in module["supported_operations"]

    invoke("--format", "json", "resource", "gnn://documentation/grammar")
    grammar = json.loads(capsys.readouterr().out)
    assert grammar["uri"] == "gnn://documentation/grammar"
    document = grammar["content"]
    assert document["success"]
    assert document["doc_name"] == "grammar"
    assert "GNN" in document["content"]
    assert "::=" in document["content"] or "=" in document["content"]

    invoke("--format", "json", "info", "get_gnn_module_info")
    info = json.loads(capsys.readouterr().out)
    assert info["name"] == "get_gnn_module_info"
    assert info["usage_count"] >= 1
    assert info["execution_count"] >= 1
    invoke("--format", "json", "status")
    status = json.loads(capsys.readouterr().out)
    assert status["request_count"] >= 1
    assert status["error_count"] == 0


@pytest.mark.parametrize(
    ("arguments", "operation", "message"),
    [
        (
            ("execute", "get_gnn_documentation", "--params", "{"),
            "executing tool",
            "Invalid JSON",
        ),
        (
            ("execute", "get_gnn_documentation", "--params", "[]"),
            "executing tool",
            "JSON object",
        ),
        (
            ("execute", "get_gnn_documentation", "--validate"),
            "executing tool",
            "doc_name",
        ),
        (("info", "missing_native_tool"), "getting tool info", "missing_native_tool"),
        (("resource", "gnn://missing/resource"), "retrieving resource", "resource"),
    ],
)
def test_native_cli_rejects_invalid_invocations_as_one_machine_error(
    native_cli,
    capsys: pytest.CaptureFixture[str],
    arguments: tuple[str, ...],
    operation: str,
    message: str,
) -> None:
    invoke, _ = native_cli
    capsys.readouterr()

    with pytest.raises(SystemExit) as error:
        invoke("--format", "json", *arguments)

    assert error.value.code == 1
    document = json.loads(capsys.readouterr().out)
    assert document["error"]["operation"] == operation
    assert message in document["error"]["message"]


def test_native_human_cli_explains_registered_capabilities_and_live_diagnostics(
    native_cli, caplog: pytest.LogCaptureFixture
) -> None:
    invoke, _ = native_cli
    caplog.set_level(logging.INFO, logger="mcp.cli")
    caplog.clear()
    invoke("--verbose", "list")
    listed = caplog.text
    assert "get_gnn_documentation" in listed
    assert "gnn://documentation/{doc_name}" in listed
    assert "Module: gnn.gnn" in listed

    caplog.clear()
    invoke("--verbose", "execute", "get_gnn_module_info")
    executed = caplog.text
    assert "executed successfully" in executed
    assert "parse_gnn_content" in executed
    assert re.search(r"Uses: [1-9][0-9]*", executed)
    caplog.clear()
    invoke("--verbose", "info", "get_gnn_module_info")
    info = caplog.text
    assert "Times Used:" in info
    assert "Schema:" in info
    caplog.clear()
    invoke("--verbose", "diagnostics")
    diagnostics = caplog.text
    assert "Overall Health:" in diagnostics
    assert "Health Checks:" in diagnostics
    assert "Module Status:" in diagnostics
