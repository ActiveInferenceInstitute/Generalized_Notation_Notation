#!/usr/bin/env python3
"""Interactive launch-verification regressions for gui_1/gui_2 (t-0034).

The interactive branches of the GUI 1/2 processors used to sleep a fixed
3 seconds and then unconditionally record ``launched: true``, even when the
server thread had died on startup. These regressions pin the bounded
``wait_for_server_launch`` contract: a failed launch must write a
``launch_failed`` artifact without port/url keys, a verified launch must
write the unified ``interactive_mode`` artifact, and a thread that stays
alive past the poll bound (but is not yet serving) still counts as
launched.
"""

from __future__ import annotations

import importlib
import json
import logging
import socket
import sys
import threading
import types
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from gnn.gui import runner as gui_runner

STEP_DIR_NAME = "22_gui_output"


def _test_logger() -> logging.Logger:
    logger = logging.getLogger("test_gui_launch_verification")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def _write_target(target: Path) -> None:
    """Materialize a minimal target directory for both GUIs."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.md").write_text(
        "# Test Model\n\n"
        "components:\n"
        "  - name: example_component\n"
        "    type: observation\n"
        "    states: [s1, s2]\n\n"
    )


def _load_status(output: Path, name: str) -> dict[str, Any]:
    status_file = output / STEP_DIR_NAME / name
    assert status_file.exists()
    status: dict[str, Any] = json.loads(status_file.read_text())
    return status


class _ServingHandler(BaseHTTPRequestHandler):
    """Always answer 200 so the launch probe sees a live server."""

    def do_GET(self) -> None:  # noqa: N802 - http.server interface name
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


class FakeDemo:
    """Gradio ``Blocks.launch`` stand-in for launch-thread simulation."""

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self.release = threading.Event()
        self._fail_with = fail_with

    def launch(self, **kwargs: Any) -> None:
        # The simulated bind failure is recorded and the target returns
        # quietly instead of raising: the poll detects the exited thread
        # either way, and pytest stays free of unhandled-thread warnings.
        if self._fail_with is not None:
            return
        assert self.release.wait(timeout=30), "launch thread was not released"


@pytest.fixture
def interactive_gradio() -> Iterator[None]:
    """Stub gradio with a Blocks attribute so gui_1/gui_2 go interactive."""
    gui1_processor = importlib.import_module("gnn.gui.gui_1.processor")
    gui2_processor = importlib.import_module("gnn.gui.gui_2.processor")

    original_gradio = sys.modules.get("gradio")
    stub = types.ModuleType("gradio")
    sys.modules["gradio"] = stub
    stub.Blocks = object  # type: ignore[attr-defined]
    importlib.reload(gui1_processor)
    importlib.reload(gui2_processor)
    try:
        yield
    finally:
        if original_gradio is None:
            sys.modules.pop("gradio", None)
        else:
            sys.modules["gradio"] = original_gradio
        importlib.reload(gui1_processor)
        importlib.reload(gui2_processor)


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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ServingHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


@pytest.fixture
def occupied_port() -> Iterator[int]:
    """A TCP port with a listening (but never accepting) socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


class TestGuiLaunchFailure:
    """A dead server thread must produce a launch_failed artifact."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui1_launch_failure_writes_launched_false(
        self,
        interactive_gradio: None,
        clean_server_registry: None,
        occupied_port: int,
        monkeypatch: pytest.MonkeyPatch,
        isolated_temp_dir: Any,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        gui1_processor = importlib.import_module("gnn.gui.gui_1.processor")
        monkeypatch.setattr(gui1_processor, "_GUI1_PORT", occupied_port)
        gui1_ui = importlib.import_module("gnn.gui.gui_1.ui")
        monkeypatch.setattr(
            gui1_ui,
            "build_gui",
            lambda *args, **kwargs: FakeDemo(
                fail_with=OSError("Address already in use")
            ),
        )

        result = gui1_processor.run_gui(
            target_dir=target, output_dir=output, logger=_test_logger(), headless=False
        )

        assert result is False
        status = _load_status(output, "gui_1_status.json")
        assert status["launched"] is False
        assert status["status"] == "launch_failed"
        assert status["reason"]
        assert "port" not in status
        assert "url" not in status

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui2_launch_failure_writes_launched_false(
        self,
        interactive_gradio: None,
        clean_server_registry: None,
        occupied_port: int,
        monkeypatch: pytest.MonkeyPatch,
        isolated_temp_dir: Any,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        gui2_processor = importlib.import_module("gnn.gui.gui_2.processor")
        monkeypatch.setattr(gui2_processor, "_GUI2_PORT", occupied_port)
        gui2_ui = importlib.import_module("gnn.gui.gui_2.ui")
        monkeypatch.setattr(
            gui2_ui,
            "build_visual_gui",
            lambda *args, **kwargs: FakeDemo(
                fail_with=OSError("Address already in use")
            ),
        )

        result = gui2_processor.run_gui(
            target_dir=target, output_dir=output, logger=_test_logger(), headless=False
        )

        assert result is False
        status = _load_status(output, "gui_2_status.json")
        assert status["launched"] is False
        assert status["status"] == "launch_failed"
        assert status["reason"]
        assert "port" not in status
        assert "url" not in status


class TestGuiInteractiveSuccess:
    """A verified launch must produce the unified interactive artifact."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui1_interactive_success_writes_unified_status(
        self,
        interactive_gradio: None,
        clean_server_registry: None,
        serving_http: int,
        monkeypatch: pytest.MonkeyPatch,
        isolated_temp_dir: Any,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        gui1_processor = importlib.import_module("gnn.gui.gui_1.processor")
        monkeypatch.setattr(gui1_processor, "_GUI1_PORT", serving_http)
        demo = FakeDemo()
        gui1_ui = importlib.import_module("gnn.gui.gui_1.ui")
        monkeypatch.setattr(gui1_ui, "build_gui", lambda *args, **kwargs: demo)

        try:
            result = gui1_processor.run_gui(
                target_dir=target,
                output_dir=output,
                logger=_test_logger(),
                headless=False,
            )
        finally:
            demo.release.set()

        assert result is True
        status = _load_status(output, "gui_1_status.json")
        assert status["launched"] is True
        assert status["status"] == "interactive_mode"
        assert status["reason"] == "gradio_launched"
        assert "backend_reason" in status
        assert status["port"] == serving_http
        assert status["url"] == f"http://localhost:{serving_http}"

    @pytest.mark.unit
    @pytest.mark.fast
    def test_gui2_interactive_success_writes_unified_status(
        self,
        interactive_gradio: None,
        clean_server_registry: None,
        serving_http: int,
        monkeypatch: pytest.MonkeyPatch,
        isolated_temp_dir: Any,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        gui2_processor = importlib.import_module("gnn.gui.gui_2.processor")
        monkeypatch.setattr(gui2_processor, "_GUI2_PORT", serving_http)
        demo = FakeDemo()
        gui2_ui = importlib.import_module("gnn.gui.gui_2.ui")
        monkeypatch.setattr(gui2_ui, "build_visual_gui", lambda *args, **kwargs: demo)

        try:
            result = gui2_processor.run_gui(
                target_dir=target,
                output_dir=output,
                logger=_test_logger(),
                headless=False,
            )
        finally:
            demo.release.set()

        assert result is True
        status = _load_status(output, "gui_2_status.json")
        assert status["launched"] is True
        assert status["status"] == "interactive_mode"
        assert status["reason"] == "gradio_launched"
        assert "backend_reason" in status
        assert "features" in status
        assert status["port"] == serving_http
        assert status["url"] == f"http://localhost:{serving_http}"


class TestGuiLaunchBound:
    """A thread alive past the poll bound counts as launched."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_alive_thread_past_bound_is_success(
        self,
        interactive_gradio: None,
        clean_server_registry: None,
        monkeypatch: pytest.MonkeyPatch,
        isolated_temp_dir: Any,
    ) -> None:
        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        # Grab a free port, then close it so nothing answers probes there.
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
        sock.close()

        # Module constants are read at call time; shrink the bound so the
        # test stays fast while the launch thread blocks past it.
        monkeypatch.setattr("gnn.gui.backend.SERVER_POLL_ATTEMPTS", 2)
        monkeypatch.setattr("gnn.gui.backend.SERVER_POLL_INTERVAL_S", 0.05)

        gui1_processor = importlib.import_module("gnn.gui.gui_1.processor")
        monkeypatch.setattr(gui1_processor, "_GUI1_PORT", free_port)
        demo = FakeDemo()
        gui1_ui = importlib.import_module("gnn.gui.gui_1.ui")
        monkeypatch.setattr(gui1_ui, "build_gui", lambda *args, **kwargs: demo)

        try:
            result = gui1_processor.run_gui(
                target_dir=target,
                output_dir=output,
                logger=_test_logger(),
                headless=False,
            )
        finally:
            demo.release.set()

        assert result is True
        status = _load_status(output, "gui_1_status.json")
        assert status["launched"] is True
        assert status["status"] == "interactive_mode"
        assert status["reason"] == "gradio_launched"


class TestProcessGuiInteractiveFailure:
    """process_gui aggregates an interactive launch failure as overall False."""

    @pytest.mark.unit
    @pytest.mark.fast
    def test_process_gui_interactive_failure_sets_overall_success_false(
        self,
        interactive_gradio: None,
        clean_server_registry: None,
        occupied_port: int,
        monkeypatch: pytest.MonkeyPatch,
        isolated_temp_dir: Any,
    ) -> None:
        from gnn.gui import process_gui

        target = isolated_temp_dir / "input"
        output = isolated_temp_dir / "output"
        _write_target(target)

        gui1_processor = importlib.import_module("gnn.gui.gui_1.processor")
        monkeypatch.setattr(gui1_processor, "_GUI1_PORT", occupied_port)
        gui1_ui = importlib.import_module("gnn.gui.gui_1.ui")
        monkeypatch.setattr(
            gui1_ui,
            "build_gui",
            lambda *args, **kwargs: FakeDemo(
                fail_with=OSError("Address already in use")
            ),
        )

        result = process_gui(
            target_dir=target,
            output_dir=output,
            logger=_test_logger(),
            interactive=True,
            gui_types="gui_1",
        )

        assert result is False
        summary_file = output / "gui_processing_summary.json"
        assert summary_file.exists()
        summary: dict[str, Any] = json.loads(summary_file.read_text())
        assert summary["overall_success"] is False
        assert summary["results"]["gui_1"]["success"] is False
