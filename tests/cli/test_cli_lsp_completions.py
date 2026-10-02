"""Behavior tests for the CLI LSP server: completion and schema diagnostics.

Exercises ``gnn.cli.lsp`` over the injected-transport session pattern from
``test_cli_composition.py`` (``TestLspLoopInjection``): monkeypatched
``write_message`` and reader callables, no real stdio. The CLI server path is
pygls-free by design, so these tests must run without pygls installed.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.cli import lsp as cli_lsp
from gnn.lsp.completions import (
    all_section_completions,
    model_parameter_key_completions,
)


def _session(messages: list[dict[str, Any]]) -> list[Any]:
    """Run one LSP session over injected transport callables."""
    written: list[Any] = []
    inbox = iter(messages)

    def reader() -> Any:
        try:
            return next(inbox)
        except StopIteration:
            return None

    def writer(msg: Any, *args: Any) -> None:
        written.append(msg)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cli_lsp, "write_message", writer)
        cli_lsp.run_lsp_loop(reader, writer)
    return written


VALID_MINIMAL_GNN = (
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

MODEL_PARAMETERS_GNN = "## ModelParameters\nnum_obs: 3\n"

# Missing GNNVersionAndFlags, StateSpaceBlock, and Connections.
MISSING_SECTIONS_GNN = "## GNNSection\nActInfPOMDP\n\n## ModelName\nSparse Model\n"

# Declared shape conflict (GNN-E002, line 17 -> index 16) and undeclared
# parameterization (GNN-W003, line 18 -> index 17).
SHAPE_CONFLICT_GNN = (
    "## GNNSection\n"
    "ActInfPOMDP\n"
    "\n"
    "## GNNVersionAndFlags\n"
    "GNN v1\n"
    "\n"
    "## ModelName\n"
    "Conflict\n"
    "\n"
    "## StateSpaceBlock\n"
    "s_f[2,1,type=float]\n"
    "\n"
    "## Connections\n"
    "s_f>s_f\n"
    "\n"
    "## InitialParameterization\n"
    "s_f = [[0.5, 0.5], [0.5, 0.5]]\n"
    "ghost_param = [[1.0]]\n"
    "\n"
    "## Time\n"
    "Dynamic\n"
    "\n"
    "## Footer\n"
    "End.\n"
)


class TestCompletionOverSession:
    """textDocument/completion is served over the injected transport loop."""

    def _completion_messages(
        self, doc: str, line: int, character: int
    ) -> list[dict[str, Any]]:
        return [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {
                "jsonrpc": "2.0",
                "method": "textDocument/didOpen",
                "params": {"textDocument": {"uri": "file:///m.md", "text": doc}},
            },
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "textDocument/completion",
                "params": {
                    "textDocument": {"uri": "file:///m.md"},
                    "position": {"line": line, "character": character},
                },
            },
            {"jsonrpc": "2.0", "id": 4, "method": "shutdown"},
            {"jsonrpc": "2.0", "method": "exit"},
        ]

    def test_completion_request_returns_items_not_method_not_found(self) -> None:
        written = _session(self._completion_messages(VALID_MINIMAL_GNN, 0, 3))
        responses = [m for m in written if m.get("id") == 3]
        assert len(responses) == 1
        response = responses[0]
        assert "error" not in response
        result = response["result"]
        assert result["isIncomplete"] is False
        labels = [item["label"] for item in result["items"]]
        assert "GNNSection" in labels
        assert labels == [item["label"] for item in all_section_completions()]

    def test_completion_inside_model_parameters_uses_opened_document(self) -> None:
        written = _session(self._completion_messages(MODEL_PARAMETERS_GNN, 1, 4))
        response = next(m for m in written if m.get("id") == 3)
        labels = [item["label"] for item in response["result"]["items"]]
        assert labels == [item["label"] for item in model_parameter_key_completions()]
        assert "num_hidden_states" in labels

    def test_initialize_still_advertises_completion_provider(self) -> None:
        written = _session(
            [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
                {"jsonrpc": "2.0", "method": "exit"},
            ]
        )
        caps = written[0]["result"]["capabilities"]
        assert caps["completionProvider"] == {
            "resolveProvider": False,
            "triggerCharacters": ["."],
        }

    def test_handle_completion_without_documents_defaults_to_sections(self) -> None:
        response = cli_lsp.handle_completion(
            5,
            {
                "textDocument": {"uri": "file:///x.md"},
                "position": {"line": 0, "character": 0},
            },
        )
        assert response["id"] == 5
        labels = [item["label"] for item in response["result"]["items"]]
        assert labels == [item["label"] for item in all_section_completions()]


class TestSchemaBackedDiagnostics:
    """didOpen publishes gnn.schema-derived diagnostics, not just brace checks."""

    def _did_open(self, uri: str, text: str) -> list[Any]:
        return _session(
            [
                {
                    "jsonrpc": "2.0",
                    "method": "textDocument/didOpen",
                    "params": {"textDocument": {"uri": uri, "text": text}},
                },
                {"jsonrpc": "2.0", "method": "exit"},
            ]
        )

    def test_did_open_missing_sections_publishes_schema_diagnostics(self) -> None:
        written = self._did_open("file:///sparse.md", MISSING_SECTIONS_GNN)
        assert written[0]["method"] == "textDocument/publishDiagnostics"
        diagnostics = written[0]["params"]["diagnostics"]
        missing = [d for d in diagnostics if "Missing required section" in d["message"]]
        assert len(missing) == 3
        assert {d["severity"] for d in missing} == {1}
        assert {d["source"] for d in missing} == {"gnn"}
        messages = " ".join(d["message"] for d in missing)
        assert "GNNVersionAndFlags" in messages
        assert "StateSpaceBlock" in messages
        assert "Connections" in messages

    def test_valid_document_publishes_no_diagnostics(self) -> None:
        written = self._did_open("file:///ok.md", VALID_MINIMAL_GNN)
        assert written[0]["params"]["diagnostics"] == []

    def test_brace_only_text_still_triggers_brace_diagnostic(self) -> None:
        written = self._did_open("file:///b.md", "{ open")
        diagnostics = written[0]["params"]["diagnostics"]
        assert diagnostics[0]["message"] == "Missing closing brace '}'"
        assert diagnostics[0]["severity"] == 1
        assert diagnostics[0]["source"] == "gnn"


class TestDiagnoseTextDelegation:
    """diagnose_text maps GNNParseError objects to the pygls diagnostic shape."""

    def test_shape_mismatch_and_undeclared_parameterization(self) -> None:
        diagnostics = cli_lsp.diagnose_text(SHAPE_CONFLICT_GNN)
        assert len(diagnostics) == 2
        e002, w003 = diagnostics
        assert "GNN-E002" in e002["message"]
        assert e002["severity"] == 1
        assert e002["source"] == "gnn"
        assert e002["range"]["start"]["line"] == 16
        assert "GNN-W003" in w003["message"]
        assert w003["severity"] == 2
        assert w003["source"] == "gnn"
        assert w003["range"]["start"]["line"] == 17

    def test_empty_text_yields_no_diagnostics(self) -> None:
        assert cli_lsp.diagnose_text("") == []
        assert cli_lsp.diagnose_text("   \n\t\n") == []

    def test_missing_section_diagnostic_shape(self) -> None:
        diagnostics = cli_lsp.diagnose_text(MISSING_SECTIONS_GNN)
        missing = [d for d in diagnostics if "Missing required section" in d["message"]]
        assert missing
        for d in missing:
            assert set(d) == {"range", "severity", "message", "source"}
            # E001 carries no line info -> line 1 -> 0-indexed line 0.
            assert d["range"]["start"]["line"] == 0
