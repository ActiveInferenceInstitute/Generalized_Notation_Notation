"""Behavior tests for ``gnn.website.serve`` (loopback HTTP serving of the
generated pipeline output tree).

Covers: static serving of ``20_website_output/`` and ``22_gui_output/``,
live-reload HTML injection plus the ``/_livereload`` version endpoint (and its
absence when disabled), the loopback-only bind guard, port-in-use mapping to
``PortInUseError``, and output-root validation. All servers bind ephemeral
port 0 and are shut down in ``finally``.
"""

from __future__ import annotations

import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.website.serve import (
    LoopbackViolationError,
    OutputRootNotFoundError,
    PortInUseError,
    WebsiteServer,
)


def _write_site(root: Path) -> None:
    """Create a minimal output tree with website and GUI HTML files."""
    site = root / "20_website_output"
    site.mkdir()
    (site / "index.html").write_text(
        "<html><body><h1>Website Index</h1></body></html>", encoding="utf-8"
    )
    gui = root / "22_gui_output"
    gui.mkdir()
    (gui / "navigation.html").write_text(
        "<html><body><nav>GUI Navigation</nav></body></html>", encoding="utf-8"
    )


class TestStaticServing:
    """The server serves the generated output tree over loopback HTTP."""

    def test_serves_website_and_gui_trees(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0)
        server.start()
        try:
            with urllib.request.urlopen(server.url + "20_website_output/") as resp:
                assert resp.status == 200
                assert (
                    resp.read().decode("utf-8")
                    == "<html><body><h1>Website Index</h1></body></html>"
                )
            with urllib.request.urlopen(
                server.url + "22_gui_output/navigation.html"
            ) as resp:
                assert resp.status == 200
                assert (
                    resp.read().decode("utf-8")
                    == "<html><body><nav>GUI Navigation</nav></body></html>"
                )
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(server.url + "missing/path.html")
            assert excinfo.value.code == 404
        finally:
            server.shutdown()

    def test_url_and_bound_port_reflect_real_binding(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0)
        server.start()
        try:
            assert server.bound_port != 0
            assert server.url == f"http://127.0.0.1:{server.bound_port}/"
            assert server.landing_url == server.url + "20_website_output/"
        finally:
            server.shutdown()

    def test_landing_url_falls_back_without_website_dir(self, tmp_path: Path) -> None:
        (tmp_path / "00_pipeline_summary").mkdir()
        server = WebsiteServer(tmp_path, port=0)
        server.start()
        try:
            assert server.landing_url == server.url
        finally:
            server.shutdown()


class TestLiveReload:
    """Live reload injects a poller script and exposes ``/_livereload``."""

    def test_html_injection_before_body_close(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0, live_reload=True)
        server.start()
        try:
            with urllib.request.urlopen(server.landing_url) as resp:
                assert resp.headers["Content-Type"] == "text/html; charset=utf-8"
                body = resp.read().decode("utf-8")
            assert "/_livereload" in body
            assert "<script>" in body
            assert body.index("/_livereload") < body.rindex("</body>")
        finally:
            server.shutdown()

    def test_livereload_endpoint_returns_hex_version(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0, live_reload=True)
        server.start()
        try:
            with urllib.request.urlopen(server.url + "_livereload") as resp:
                assert resp.status == 200
                assert resp.headers["Content-Type"] == "text/plain"
                version = resp.read().decode("utf-8")
            assert len(version) == 64
            assert all(char in "0123456789abcdef" for char in version)
        finally:
            server.shutdown()

    def test_livereload_version_changes_on_touch(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0, live_reload=True)
        server.start()
        try:
            with urllib.request.urlopen(server.url + "_livereload") as resp:
                before = resp.read().decode("utf-8")
            index = tmp_path / "20_website_output" / "index.html"
            future = time.time() + 1000
            os.utime(index, (future, future))
            with urllib.request.urlopen(server.url + "_livereload") as resp:
                after = resp.read().decode("utf-8")
            assert after != before
        finally:
            server.shutdown()

    def test_livereload_disabled_keeps_html_and_endpoint_absent(
        self, tmp_path: Path
    ) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0, live_reload=False)
        server.start()
        try:
            with urllib.request.urlopen(server.landing_url) as resp:
                body = resp.read().decode("utf-8")
            assert "/_livereload" not in body
            with pytest.raises(urllib.error.HTTPError) as excinfo:
                urllib.request.urlopen(server.url + "_livereload")
            assert excinfo.value.code == 404
        finally:
            server.shutdown()


class TestLoopbackGuard:
    """Non-loopback bind hosts are refused by name; localhost aliases 127.0.0.1."""

    def test_non_loopback_hosts_refused(self, tmp_path: Path) -> None:
        for host in ("0.0.0.0", "192.168.1.10"):
            server = WebsiteServer(tmp_path, port=0, host=host)
            with pytest.raises(LoopbackViolationError) as excinfo:
                server.start()
            assert host in str(excinfo.value)

    def test_localhost_binds_loopback(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        server = WebsiteServer(tmp_path, port=0, host="localhost")
        server.start()
        try:
            assert "127.0.0.1" in server.url
        finally:
            server.shutdown()


class TestPortInUse:
    """An occupied port maps the bind failure to ``PortInUseError``.

    Calibrated on this machine: a second bind against a listening socket fails
    with ``OSError`` errno 48 (Address already in use), including for
    ``ThreadingHTTPServer`` with its default ``allow_reuse_address``.
    """

    def test_bind_conflict_raises_port_in_use(self, tmp_path: Path) -> None:
        _write_site(tmp_path)
        holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            holder.bind(("127.0.0.1", 0))
            holder.listen(1)
            port = holder.getsockname()[1]
            server = WebsiteServer(tmp_path, port=port)
            with pytest.raises(PortInUseError) as excinfo:
                server.start()
            assert str(port) in str(excinfo.value)
        finally:
            holder.close()


class TestRootValidation:
    """A missing output root fails loudly with the exact path."""

    def test_missing_output_root_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "does-not-exist"
        server = WebsiteServer(missing, port=0)
        with pytest.raises(OutputRootNotFoundError) as excinfo:
            server.start()
        assert str(missing) in str(excinfo.value)
