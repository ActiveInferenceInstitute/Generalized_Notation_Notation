"""Behavior tests for the ``gnn serve --surface website`` CLI wiring.

Covers: dispatch to ``gnn.website.serve.serve_website`` with the handler's
resolved defaults, the port default split (8090 website / 8000 API), loud
error mapping for ``WebsiteServerError`` subclasses, the website-only flag
guard for API surfaces, and the parser surface. All external effects are
monkeypatched at their source modules, so nothing binds a socket here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gnn.cli as cli
from gnn.website.serve import LoopbackViolationError, OutputRootNotFoundError


class TestServeWebsiteDispatch:
    """``_cmd_serve`` routes the website surface to serve_website."""

    def test_serve_website_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.website import serve as serve_mod

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(serve_mod, "serve_website", record)
        args = argparse.Namespace(
            surface="website",
            host="127.0.0.1",
            port=8090,
            root=str(tmp_path),
            live_reload=True,
            verbose=False,
        )
        assert cli._cmd_serve(args) == cli.EXIT_SUCCESS
        assert calls == [
            (
                (Path(tmp_path),),
                {
                    "port": 8090,
                    "open_browser": False,
                    "live_reload": True,
                    "host": "127.0.0.1",
                },
            )
        ]

    def test_website_port_default_resolution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from gnn.website import serve as serve_mod

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(serve_mod, "serve_website", record)
        args = argparse.Namespace(
            surface="website",
            host="127.0.0.1",
            port=None,
            root=str(tmp_path),
            live_reload=False,
            verbose=False,
        )
        assert cli._cmd_serve(args) == cli.EXIT_SUCCESS
        assert calls[0][1]["port"] == 8090

    def test_api_port_default_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from gnn.api import app as api_app

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(api_app, "start_server", record)
        args = argparse.Namespace(
            surface="runs",
            host="127.0.0.1",
            port=None,
            root=None,
            live_reload=False,
            verbose=False,
        )
        assert cli._cmd_serve(args) == cli.EXIT_SUCCESS
        assert calls == [((), {"host": "127.0.0.1", "port": 8000})]


class TestServeWebsiteErrors:
    """Website server failures map to EXIT_ERROR with a loud message."""

    def test_website_loopback_refusal_via_cli(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from gnn.website import serve as serve_mod

        def record(*args: Any, **kwargs: Any) -> None:
            raise LoopbackViolationError("Refusing non-loopback host 0.0.0.0")

        monkeypatch.setattr(serve_mod, "serve_website", record)
        args = argparse.Namespace(
            surface="website",
            host="0.0.0.0",
            port=8090,
            root=str(tmp_path),
            live_reload=False,
            verbose=False,
        )
        assert cli._cmd_serve(args) == cli.EXIT_ERROR
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "0.0.0.0" in captured.out

    def test_missing_root_is_loud(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from gnn.website import serve as serve_mod

        def record(*args: Any, **kwargs: Any) -> None:
            raise OutputRootNotFoundError(
                f"Output root not found: {tmp_path / 'missing'}"
            )

        monkeypatch.setattr(serve_mod, "serve_website", record)
        args = argparse.Namespace(
            surface="website",
            host="127.0.0.1",
            port=8090,
            root=str(tmp_path / "missing"),
            live_reload=False,
            verbose=False,
        )
        assert cli._cmd_serve(args) == cli.EXIT_ERROR
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "missing" in captured.out

    def test_website_only_flags_rejected_for_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from gnn.api import app as api_app

        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def record(*args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

        monkeypatch.setattr(api_app, "start_server", record)
        args = argparse.Namespace(
            surface="runs",
            host="127.0.0.1",
            port=8000,
            root=None,
            live_reload=True,
            verbose=False,
        )
        assert cli._cmd_serve(args) == cli.EXIT_ERROR
        assert calls == []
        captured = capsys.readouterr()
        assert "❌" in captured.out
        assert "live" in captured.out.lower()


class TestServeParserSurface:
    """The parser accepts the website surface and its dedicated flags."""

    def test_parser_accepts_website_flags(self) -> None:
        parser = cli.build_parser()
        parsed = parser.parse_args(
            [
                "serve",
                "--surface",
                "website",
                "--port",
                "8090",
                "--root",
                "/tmp/x",
                "--live-reload",
            ]
        )
        assert parsed.surface == "website"
        assert parsed.port == 8090
        assert parsed.root == "/tmp/x"
        assert parsed.live_reload is True

    def test_serve_defaults_unchanged_for_api(self) -> None:
        parser = cli.build_parser()
        parsed = parser.parse_args(["serve"])
        assert parsed.surface == "runs"
        assert parsed.port is None
