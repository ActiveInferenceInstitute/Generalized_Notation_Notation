"""Behavior tests for GNN LSP completions and schema-derived diagnostics.

The vocabulary module (``gnn.lsp.completions``) is pygls-free and always
testable. pygls-server integration (feature registration, published
diagnostics) follows the availability conventions of ``test_lsp_server.py``:
pygls-present paths run the real server; pygls-absent paths assert the
graceful contract (``create_server() is None``). The suite never skips —
the zero-skip contract lives in ``tests/test_zero_skip_contracts.py``.
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.lsp import PYGLS_AVAILABLE, create_server
from gnn.lsp.completions import (
    MODEL_PARAMETER_KEYS,
    all_section_completions,
    completion_context,
    context_completions,
    dtype_completions,
    gnn_section_value_completions,
    model_parameter_key_completions,
)
from gnn.schemas.section_contract import (
    CANONICAL_GNN_SECTIONS,
    OPTIONAL_SECTIONS,
    REQUIRED_SECTIONS,
)


def _assert_pygls_graceful_when_absent() -> bool:
    """pygls-absent environments assert the graceful contract instead of
    skipping (skip APIs are banned by tests/test_zero_skip_contracts.py):
    ``create_server()`` must return None when pygls is unavailable.

    Returns True when the caller should stop after the graceful assertions.
    """
    if not PYGLS_AVAILABLE:
        assert create_server() is None
        return True
    return False


# A fully-valid minimal GNN document: all parse-level required sections,
# declared + connected variables, no parameterization to cross-check.
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

# Minimal GNN (modeled on the input/gnn_files exemplars) with a declared-shape
# conflict (GNN-E002), an undeclared parameterization (GNN-W003), and one
# unknown section header. Pinned line indexes: the s_f assignment ends on
# line 18 (index 17), ghost_param on line 19 (index 18), `## NotASection` is
# line 27 (index 26).
DIAGNOSTIC_PROBE_GNN = (
    "## GNNSection\n"
    "ActInfPOMDP\n"
    "\n"
    "## GNNVersionAndFlags\n"
    "GNN v1\n"
    "\n"
    "## ModelName\n"
    "Shape Conflict Probe\n"
    "\n"
    "## StateSpaceBlock\n"
    "s_f[2,1,type=float]\n"
    "s_x[2,1,type=float]\n"
    "\n"
    "## Connections\n"
    "s_f>s_x\n"
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
    "\n"
    "## NotASection\n"
    "junk\n"
)


class TestSectionVocabulary:
    """Section completions mirror gnn.schemas.section_contract exactly."""

    def test_all_thirteen_canonical_sections_offered_in_declared_order(self) -> None:
        items = all_section_completions()
        assert [item["label"] for item in items] == list(CANONICAL_GNN_SECTIONS)
        assert len(items) == 13

    def test_section_detail_matches_compliance_tables(self) -> None:
        for item in all_section_completions():
            label = item["label"]
            if label in REQUIRED_SECTIONS:
                assert item["detail"] == "Required GNN section"
            else:
                assert label in OPTIONAL_SECTIONS
                assert item["detail"] == "Optional GNN section"

    def test_items_carry_the_full_lsp_item_shape(self) -> None:
        for item in all_section_completions():
            assert set(item) == {"label", "kind", "detail", "insert_text"}
            assert item["insert_text"] == item["label"]


class TestValueVocabularies:
    """Value/key/dtype vocabularies come from their canonical sources."""

    def test_gnn_section_values(self) -> None:
        values = [item["label"] for item in gnn_section_value_completions()]
        assert values == [
            "ActInfPOMDP",
            "ActInfPOMDP_MultiAgent",
            "ActInfContinuous",
        ]

    def test_model_parameter_keys(self) -> None:
        keys = [item["label"] for item in model_parameter_key_completions()]
        assert keys == list(MODEL_PARAMETER_KEYS)
        assert "num_hidden_states" in keys

    def test_dtype_values_match_schema_enum(self) -> None:
        from gnn.schema import GNN_MODEL_SCHEMA

        enum = GNN_MODEL_SCHEMA["properties"]["state_space"]["items"]["properties"][
            "dtype"
        ]["enum"]
        assert [item["label"] for item in dtype_completions()] == enum
        assert set(enum) == {"float", "int", "bool"}


class TestContextSelection:
    """context_completions picks the vocabulary by simple context rules."""

    def test_header_prefix_yields_sections(self) -> None:
        labels = [item["label"] for item in context_completions("## Stat")]
        assert "StateSpaceBlock" in labels
        assert labels == list(CANONICAL_GNN_SECTIONS)

    def test_bare_header_marker_yields_sections_not_section_values(self) -> None:
        # Cursor truncated exactly after "## " on a GNNSection header line:
        # rule 1 must win over the enclosing-section flag.
        labels = [
            item["label"] for item in context_completions("## ", in_gnn_section=True)
        ]
        assert labels == list(CANONICAL_GNN_SECTIONS)

    def test_model_parameters_context_yields_parameter_keys(self) -> None:
        labels = [
            item["label"]
            for item in context_completions("num_", in_model_parameters=True)
        ]
        assert "num_hidden_states" in labels
        assert labels == [item["label"] for item in model_parameter_key_completions()]

    def test_type_context_yields_dtypes(self) -> None:
        labels = [item["label"] for item in context_completions("s_f[2,1,type=flo")]
        assert labels == ["float", "int", "bool"]

    def test_gnn_section_context_yields_values(self) -> None:
        labels = [
            item["label"] for item in context_completions("ActInf", in_gnn_section=True)
        ]
        assert labels == ["ActInfPOMDP", "ActInfPOMDP_MultiAgent", "ActInfContinuous"]

    def test_default_context_yields_sections(self) -> None:
        labels = [item["label"] for item in context_completions("plain prose")]
        assert labels == list(CANONICAL_GNN_SECTIONS)


class TestCompletionContext:
    """completion_context derives the cursor line prefix and section flags."""

    def test_header_line_yields_prefix_with_own_section_flag(self) -> None:
        text = "## ModelParameters\nnum_obs: 3\n"
        # The closest `## ` header at or before the cursor line includes the
        # header line itself; rule 1 still wins for `## `-prefixed prefixes.
        assert completion_context(text, 0, 5) == ("## Mo", True, False)

    def test_line_inside_model_parameters_sets_flag(self) -> None:
        text = "## ModelParameters\nnum_obs: 3\n"
        assert completion_context(text, 1, 4) == ("num_", True, False)

    def test_line_inside_gnn_section_sets_flag(self) -> None:
        text = "## GNNSection\nActInfPOMDP\n"
        assert completion_context(text, 1, 6) == ("ActInf", False, True)

    def test_closest_preceding_header_wins(self) -> None:
        text = "## GNNSection\nActInfPOMDP\n\n## ModelParameters\nnum_obs: 3"
        assert completion_context(text, 4, 3) == ("num", True, False)

    def test_cursor_on_header_line_resets_flags(self) -> None:
        text = "## ModelParameters\nnum_obs: 3\n## Footer\n"
        assert completion_context(text, 2, 4) == ("## F", False, False)

    def test_out_of_range_line_keeps_last_header_context(self) -> None:
        text = "## ModelParameters\nnum_obs: 3"
        assert completion_context(text, 99, 0) == ("", True, False)

    def test_empty_document(self) -> None:
        assert completion_context("", 0, 0) == ("", False, False)


class _CapturingServer:
    """Stand-in exposing the pygls 1.x publish API."""

    def __init__(self) -> None:
        self.published: list[tuple[str, list[Any]]] = []

    def publish_diagnostics(self, uri: str, diagnostics: list[Any]) -> None:
        self.published.append((uri, list(diagnostics)))


class _CapturingServerV2:
    """Stand-in exposing only the pygls 2.x publish API."""

    def __init__(self) -> None:
        self.params: list[Any] = []

    def text_document_publish_diagnostics(self, params: Any) -> None:
        self.params.append(params)


class TestPyglsServerCompletion:
    """The pygls server registers and serves the completion feature."""

    def test_features_honest_and_feature_registered(self) -> None:
        import gnn.lsp as lsp

        assert lsp.FEATURES["completion"] is True
        if _assert_pygls_graceful_when_absent():
            return
        server = lsp.create_server()
        assert server is not None
        fm = getattr(server.protocol, "fm", None)
        if fm is not None:
            assert "textDocument/completion" in fm.features
        else:
            # pygls internals differ across versions; assert the behavioral
            # contract (the adapter serves the feature) instead of skipping.
            from lsprotocol.types import Position

            from gnn.lsp import _get_completions

            items = _get_completions("## Stat", Position(line=0, character=7))
            assert [item.label for item in items]

    def test_completion_adapter_returns_items_for_header_prefix(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import Position

        from gnn.lsp import _get_completions

        items = _get_completions("## Stat", Position(line=0, character=7))
        assert items is not None
        labels = [item.label for item in items]
        assert "StateSpaceBlock" in labels
        assert len(labels) == 13

    def test_completion_adapter_reads_section_context_from_document(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import Position

        from gnn.lsp import _get_completions

        text = "## GNNSection\nActInfPOMDP\n\n## ModelParameters\nnum_obs: 3"
        items = _get_completions(text, Position(line=4, character=4))
        assert items is not None
        labels = [item.label for item in items]
        assert "num_hidden_states" in labels


class TestPyglsDiagnostics:
    """Published diagnostics include E002/W003 and unknown-section warnings."""

    def _published(self, text: str) -> list[Any]:
        from gnn.lsp import _publish_diagnostics

        server = _CapturingServer()
        _publish_diagnostics(server, "file:///probe.md", text)
        assert server.published, "diagnostics were never published"
        return server.published[0][1]

    def test_valid_document_publishes_no_diagnostics(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        assert self._published(VALID_MINIMAL_GNN) == []

    def test_unknown_section_header_warned_once_and_valid_headers_clean(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import DiagnosticSeverity

        diagnostics = self._published(DIAGNOSTIC_PROBE_GNN)
        flagged = [
            d for d in diagnostics if d.message.startswith("Unknown section header")
        ]
        assert [d.message for d in flagged] == ["Unknown section header 'NotASection'"]
        assert flagged[0].severity == DiagnosticSeverity.Warning
        assert flagged[0].source == "gnn"
        assert flagged[0].range.start.line == 26

    def test_matrix_shape_mismatch_published_as_error(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import DiagnosticSeverity

        diagnostics = self._published(DIAGNOSTIC_PROBE_GNN)
        e002 = [d for d in diagnostics if "GNN-E002" in d.message]
        assert len(e002) == 1
        assert e002[0].severity == DiagnosticSeverity.Error
        assert "s_f" in e002[0].message
        assert e002[0].range.start.line == 17

    def test_undeclared_parameterization_published_as_warning(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import DiagnosticSeverity

        diagnostics = self._published(DIAGNOSTIC_PROBE_GNN)
        w003 = [d for d in diagnostics if "GNN-W003" in d.message]
        assert len(w003) == 1
        assert w003[0].severity == DiagnosticSeverity.Warning
        assert "ghost_param" in w003[0].message
        assert w003[0].range.start.line == 18

    def test_publish_uses_pygls_2x_api_when_present(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import PublishDiagnosticsParams

        from gnn.lsp import _publish_diagnostics

        server = _CapturingServerV2()
        _publish_diagnostics(server, "file:///v2.md", VALID_MINIMAL_GNN)
        assert len(server.params) == 1
        assert isinstance(server.params[0], PublishDiagnosticsParams)
        assert server.params[0].uri == "file:///v2.md"
        assert server.params[0].diagnostics == []

    def test_publish_falls_back_to_pygls_1x_api(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from gnn.lsp import _publish_diagnostics

        server = _CapturingServer()
        _publish_diagnostics(server, "file:///v1.md", VALID_MINIMAL_GNN)
        assert server.published == [("file:///v1.md", [])]


class TestPyglsDidChange:
    """The didChange feature re-publishes diagnostics through the real dispatch path."""

    @staticmethod
    def _drive(gen: Any) -> Any:
        """Consume a pygls builtin generator the way the JSON-RPC dispatcher
        does: invoke each yielded (handler, args, kwargs) and send its result
        back into the generator, so the full dispatch path runs end to end."""
        try:
            item = next(gen)
            while True:
                handler, args, kwargs = item
                item = gen.send(handler(*args, **(kwargs or {})))
        except StopIteration as stop:
            return stop.value

    @staticmethod
    def _initialized_capturing_server(
        monkeypatch: Any,
    ) -> tuple[Any, list[tuple[str, list[Any]]]]:
        """Real pygls server with an initialized workspace plus a capture list
        wired in place of ``gnn.lsp._publish_to_server``."""
        from lsprotocol.types import ClientCapabilities, InitializeParams

        captured: list[tuple[str, list[Any]]] = []
        monkeypatch.setattr(
            "gnn.lsp._publish_to_server",
            lambda server, uri, diagnostics: captured.append((uri, list(diagnostics))),
        )
        server = create_server()
        assert server is not None
        TestPyglsDidChange._drive(
            server.protocol.lsp_initialize(
                InitializeParams(capabilities=ClientCapabilities())
            )
        )
        return server, captured

    def test_did_change_feature_registered(self) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import TEXT_DOCUMENT_DID_CHANGE

        fm = create_server().protocol.fm
        assert TEXT_DOCUMENT_DID_CHANGE in fm.features

    def test_did_change_republishes_diagnostics(self, monkeypatch: Any) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import (
            DidChangeTextDocumentParams,
            DidOpenTextDocumentParams,
            TextDocumentContentChangeWholeDocument,
            TextDocumentItem,
            VersionedTextDocumentIdentifier,
        )

        server, captured = self._initialized_capturing_server(monkeypatch)
        uri = "file:///did-change.md"
        # didOpen puts the valid document in the workspace via the real builtin.
        self._drive(
            server.protocol.lsp_text_document__did_open(
                DidOpenTextDocumentParams(
                    text_document=TextDocumentItem(
                        uri=uri,
                        language_id="gnn",
                        version=1,
                        text=VALID_MINIMAL_GNN,
                    )
                )
            )
        )
        captured.clear()  # isolate the didChange publish from the didOpen publish
        self._drive(
            server.protocol.lsp_text_document__did_change(
                DidChangeTextDocumentParams(
                    text_document=VersionedTextDocumentIdentifier(uri=uri, version=2),
                    content_changes=[
                        TextDocumentContentChangeWholeDocument(text=VALID_MINIMAL_GNN)
                    ],
                )
            )
        )
        assert captured == [(uri, [])]

    def test_did_change_reports_new_errors(self, monkeypatch: Any) -> None:
        if _assert_pygls_graceful_when_absent():
            return
        from lsprotocol.types import (
            DiagnosticSeverity,
            DidChangeTextDocumentParams,
            DidOpenTextDocumentParams,
            TextDocumentContentChangeWholeDocument,
            TextDocumentItem,
            VersionedTextDocumentIdentifier,
        )

        server, captured = self._initialized_capturing_server(monkeypatch)
        uri = "file:///did-change-broken.md"
        self._drive(
            server.protocol.lsp_text_document__did_open(
                DidOpenTextDocumentParams(
                    text_document=TextDocumentItem(
                        uri=uri,
                        language_id="gnn",
                        version=1,
                        text=VALID_MINIMAL_GNN,
                    )
                )
            )
        )
        captured.clear()
        # The edit replaces the whole document with the broken probe document.
        self._drive(
            server.protocol.lsp_text_document__did_change(
                DidChangeTextDocumentParams(
                    text_document=VersionedTextDocumentIdentifier(uri=uri, version=2),
                    content_changes=[
                        TextDocumentContentChangeWholeDocument(
                            text=DIAGNOSTIC_PROBE_GNN
                        )
                    ],
                )
            )
        )
        assert captured and captured[0][0] == uri
        diagnostics = captured[0][1]
        assert diagnostics, "didChange published no diagnostics for the broken document"
        errors = [d for d in diagnostics if d.severity == DiagnosticSeverity.Error]
        assert errors, f"expected an error-severity diagnostic: {diagnostics}"
        assert any("GNN-E002" in d.message for d in errors)
