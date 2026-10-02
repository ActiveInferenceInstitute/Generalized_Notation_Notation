#!/usr/bin/env python3
"""GUI 3 loader/wrapper regressions for wave t-0034 (G3-1/G3-2/G3-4/G3-18).

Covers three behaviors:

- Loader honesty: starter-content discovery delegates to the shared
  ``load_first_markdown`` helper, so non-UTF8 files fall back to the default
  template without crashing, ``actinf_pomdp_agent.md`` wins over other
  top-level files, and a recursive ``**/*.md`` walk finds nested files
  (pre-fix code only globbed the top level).
- Wrapper honesty: the ``gui_3`` wrapper emits ``port``/``url`` ONLY when the
  run was interactive, succeeded, and a Gradio backend was actually present.
- Signature honesty: ``run_gui`` no longer accepts the unused ``verbose``
  parameter.
"""

from __future__ import annotations

import http.server
import importlib
import json
import logging
import socket
import sys
import threading
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from gnn.gui import runner as gui_runner
from gnn.gui.gui_3 import gui_3
from gnn.gui.gui_3.processor import run_gui as run_gui_3

STEP_DIR_NAME = "22_gui_output"
DEFAULT_TEMPLATE_HEADER = "# GNN Example: Active Inference POMDP Agent (Design Studio)"


def _test_logger() -> logging.Logger:
    return logging.getLogger("test_gui3_loader_and_wrapper")


def _write_target(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)


class _OkHandler(http.server.BaseHTTPRequestHandler):
    """Answer every GET with a bare 200 so the launch probe succeeds."""

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


class _BlockingDemo:
    """Fake demo whose launch blocks until released (simulates a live server)."""

    def __init__(self, release: threading.Event) -> None:
        self._release = release

    def launch(self, **kwargs: Any) -> Any:
        self._release.wait(timeout=30)
        return None


class _RaisingDemo:
    """Fake demo whose launch fails immediately (simulates a port clash).

    The target returns quietly rather than raising: the launch poll detects
    the exited thread either way, and pytest stays free of
    unhandled-thread-exception warnings.
    """

    def launch(self, **kwargs: Any) -> Any:
        return None


@pytest.fixture
def interactive_gradio() -> Iterator[None]:
    """Stub gradio to a module exposing Blocks so gui_3 enters interactive mode."""
    gui3_processor = importlib.import_module("gnn.gui.gui_3.processor")

    original_gradio = sys.modules.get("gradio")
    stub = types.ModuleType("gradio")
    stub.Blocks = object  # type: ignore[attr-defined]
    sys.modules["gradio"] = stub
    importlib.reload(gui3_processor)
    try:
        yield
    finally:
        if original_gradio is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = original_gradio
        importlib.reload(gui3_processor)


@pytest.fixture
def clean_server_registry() -> Iterator[None]:
    """Isolate the runner's global server-thread registry per test."""
    gui_runner.clear_launched_server_threads()
    try:
        yield
    finally:
        gui_runner.clear_launched_server_threads()


@pytest.fixture
def serving_http() -> Iterator[int]:
    """A real HTTP server answering 200 on an ephemeral 127.0.0.1 port."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
    port = int(server.server_address[1])
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield port
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def occupied_port() -> Iterator[int]:
    """A listening socket bound to an ephemeral port (connection succeeds,
    but no HTTP response ever comes)."""
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = int(blocker.getsockname()[1])
    try:
        yield port
    finally:
        blocker.close()


class TestGui3Loader:
    """Starter-content discovery via load_first_markdown (headless runs)."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_non_utf8_starter_falls_back_without_crash(
        self, isolated_temp_dir: Any
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)
        (target / "actinf_pomdp_agent.md").write_bytes(b"\xff\xfe<binary>")

        assert run_gui_3(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            headless=True,
        )

        exported = output / STEP_DIR_NAME / "designed_model_gui_3.md"
        assert exported.is_file()
        content = exported.read_text(encoding="utf-8")
        assert content.startswith(DEFAULT_TEMPLATE_HEADER)

    @pytest.mark.unit
    @pytest.mark.fast
    def test_prefers_actinf_pomdp_agent_file(self, isolated_temp_dir: Any) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)
        (target / "other_model.md").write_text("OTHER-MARKER\n")
        (target / "actinf_pomdp_agent.md").write_text("POMDP-MARKER\n")

        assert run_gui_3(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            headless=True,
        )

        exported = output / STEP_DIR_NAME / "designed_model_gui_3.md"
        content = exported.read_text(encoding="utf-8")
        assert "POMDP-MARKER" in content
        assert "OTHER-MARKER" not in content

    @pytest.mark.unit
    @pytest.mark.fast
    def test_recursive_discovery_loads_nested_file(
        self, isolated_temp_dir: Any
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        nested = target / "nested" / "sub"
        nested.mkdir(parents=True)
        (nested / "model.md").write_text("NESTED-MARKER\n")

        assert run_gui_3(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            headless=True,
        )

        exported = output / STEP_DIR_NAME / "designed_model_gui_3.md"
        content = exported.read_text(encoding="utf-8")
        assert "NESTED-MARKER" in content


class TestGui3Wrapper:
    """gui_3 wrapper emits port/url only for a verified interactive launch."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_wrapper_omits_port_url_when_headless(self, isolated_temp_dir: Any) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        result = gui_3(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            headless=True,
        )

        assert result["success"] is True
        assert "port" not in result
        assert "url" not in result

    @pytest.mark.unit
    @pytest.mark.fast
    def test_wrapper_omits_port_url_when_launch_fails(
        self,
        isolated_temp_dir: Any,
        monkeypatch: pytest.MonkeyPatch,
        interactive_gradio: None,
        clean_server_registry: None,
        occupied_port: int,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        gui3_processor = importlib.import_module("gnn.gui.gui_3.processor")
        ui_designer = importlib.import_module("gnn.gui.gui_3.ui_designer")
        monkeypatch.setattr(gui3_processor, "_GUI3_PORT", occupied_port)
        monkeypatch.setattr(
            ui_designer,
            "build_design_studio",
            lambda markdown_text, export_path, logger: _RaisingDemo(),
        )

        result = gui_3(target_dir=target, output_dir=output, logger=_test_logger())

        assert result["success"] is False
        assert "port" not in result
        assert "url" not in result

        status = json.loads(
            (output / STEP_DIR_NAME / "design_studio_status.json").read_text()
        )
        assert status["launched"] is False
        assert status["status"] == "launch_failed"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_wrapper_includes_port_url_when_launched(
        self,
        isolated_temp_dir: Any,
        monkeypatch: pytest.MonkeyPatch,
        interactive_gradio: None,
        clean_server_registry: None,
        serving_http: int,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        release = threading.Event()
        try:
            gui3_processor = importlib.import_module("gnn.gui.gui_3.processor")
            ui_designer = importlib.import_module("gnn.gui.gui_3.ui_designer")
            monkeypatch.setattr(gui3_processor, "_GUI3_PORT", serving_http)
            monkeypatch.setattr(
                ui_designer,
                "build_design_studio",
                lambda markdown_text, export_path, logger: _BlockingDemo(release),
            )

            result = gui_3(target_dir=target, output_dir=output, logger=_test_logger())

            assert result["success"] is True
            assert result["port"] == serving_http
            assert result["url"] == f"http://localhost:{serving_http}"
        finally:
            release.set()

    @pytest.mark.unit
    @pytest.mark.fast
    def test_run_gui_rejects_verbose_kwarg(self, isolated_temp_dir: Any) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        with pytest.raises(TypeError):
            run_gui_3(
                target_dir=target,
                output_dir=output,
                logger=_test_logger(),
                headless=True,
                verbose=True,
            )
