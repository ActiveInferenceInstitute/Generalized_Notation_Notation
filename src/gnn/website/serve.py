"""Serve the generated pipeline output tree over loopback HTTP.

Purpose: expose the numbered pipeline output directories written by the
pipeline (``20_website_output/``, ``22_gui_output/``, ``00_pipeline_summary/``,
...) over a loopback-only HTTP server so the generated website can be browsed
locally without a ``file://`` URL.

Precedents mirrored here:
- ``src/gnn/mcp/server_http.py`` (``MCPHTTPServer`` start/shutdown pattern:
  bind a ``ThreadingHTTPServer``, run ``serve_forever`` on a daemon thread,
  graceful ``shutdown`` + ``server_close`` from the caller).
- ``src/gnn/gui/runner.py`` (daemon-thread launch for long-lived servers).
- ``src/gnn/cli/handlers_service.py::require_secure_bind`` (loopback-only bind
  precedent; re-implemented locally so this module imports nothing from
  ``gnn`` and stays importable standalone).

Port map note: 8000 API / 8080 MCP / 7860-7862 GUIs / 5151 oxdraw / 8090
website. This module owns 8090.
"""

from __future__ import annotations

import functools
import hashlib
import ipaddress
import logging
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

logger = logging.getLogger("gnn.website.serve")

DEFAULT_WEBSITE_PORT = 8090
WEBSITE_DIR = "20_website_output"

# Cap the os.walk backing the live-reload digest so a pathological output
# tree cannot hang the poller; the digest is a cheap change signal, not an
# inventory, so truncation at the cap is acceptable.
_LIVERELOAD_WALK_CAP = 20000

_LIVE_RELOAD_SNIPPET = '<script>(function(){var base=null;setInterval(function(){fetch("/_livereload").then(function(r){return r.text();}).then(function(v){if(base===null){base=v;}else if(v!==base){location.reload();}}).catch(function(){});},1000);})();</script>'


class WebsiteServerError(Exception):
    """Base error for the website serving module."""


class PortInUseError(WebsiteServerError):
    """Raised when the requested bind address is already in use."""


class LoopbackViolationError(WebsiteServerError):
    """Raised when a non-loopback bind host is refused."""


class OutputRootNotFoundError(WebsiteServerError):
    """Raised when the configured output root does not exist."""


def _digest_hex(file_count: int, max_mtime_ns: int) -> str:
    """Return the live-reload digest over the file-count/mtime fingerprint."""
    return hashlib.sha256(f"{file_count}:{max_mtime_ns}".encode("utf-8")).hexdigest()


def _compute_livereload_digest(root: Path) -> str:
    """Hash a cheap filesystem fingerprint of ``root`` for live reload."""
    file_count = 0
    max_mtime_ns = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if file_count >= _LIVERELOAD_WALK_CAP:
                return _digest_hex(file_count, max_mtime_ns)
            path = os.path.join(dirpath, name)
            if os.path.isfile(path):
                file_count += 1
                max_mtime_ns = max(max_mtime_ns, os.stat(path).st_mtime_ns)
    return _digest_hex(file_count, max_mtime_ns)


class _WebsiteRequestHandler(SimpleHTTPRequestHandler):
    """Request handler rooted at the pipeline output root.

    ``directory`` is supplied by the caller (the output root); the
    ``live_reload`` flag toggles the ``/_livereload`` digest endpoint and the
    live-reload snippet injection into served HTML pages.
    """

    def __init__(self, *args: Any, live_reload: bool = False, **kwargs: Any) -> None:
        self._live_reload = live_reload
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Route per-request logging to the module logger at debug level."""
        logger.debug("%s %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        path_only = self.path.split("?", 1)[0].split("#", 1)[0]
        if path_only == "/_livereload":
            self._handle_livereload()
            return
        if self._live_reload:
            target = Path(self.translate_path(self.path))
            if target.is_dir():
                target = target / "index.html"
            if target.suffix == ".html" and target.is_file():
                self._serve_html_with_reload_snippet(target)
                return
        super().do_GET()

    def _handle_livereload(self) -> None:
        if not self._live_reload:
            # The endpoint must not exist when the feature is off.
            self.send_error(HTTPStatus.NOT_FOUND, "live reload is disabled")
            return
        digest = _compute_livereload_digest(Path(self.directory))
        payload = digest.encode("ascii")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_html_with_reload_snippet(self, path: Path) -> None:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, f"Cannot read {path}: {exc}")
            return
        snippet = _LIVE_RELOAD_SNIPPET.encode("utf-8")
        closing_body = payload.rfind(b"</body>")
        if closing_body >= 0:
            payload = payload[:closing_body] + snippet + payload[closing_body:]
        else:
            payload = payload + snippet
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class WebsiteServer:
    """Loopback HTTP server over a pipeline output root.

    Runs a ``ThreadingHTTPServer`` on a daemon thread: ``start`` validates the
    configuration, binds, and launches the serving thread; ``wait`` blocks
    until that thread exits; ``shutdown`` is the graceful stop and is
    idempotent when the server was never started.
    """

    def __init__(
        self,
        output_root: str | Path,
        *,
        port: int = DEFAULT_WEBSITE_PORT,
        host: str = "127.0.0.1",
        live_reload: bool = False,
    ) -> None:
        """Initialize the instance."""
        self.output_root = Path(output_root)
        self.port = port
        self.host = host
        self.live_reload = live_reload
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._bind_host: str | None = None

    @property
    def url(self) -> str:
        """Base URL of the server; always loopback, never ``localhost``."""
        host = self._bind_host if self._bind_host is not None else self.host
        if host == "localhost":
            host = "127.0.0.1"
        display = f"[{host}]" if ":" in host else host
        return f"http://{display}:{self.bound_port}/"

    @property
    def landing_url(self) -> str:
        """URL a browser should open: the website dir when it exists."""
        if (self.output_root / WEBSITE_DIR).is_dir():
            return self.url + WEBSITE_DIR + "/"
        return self.url

    @property
    def bound_port(self) -> int:
        """Actual bound port (``port=0`` resolves to the ephemeral port)."""
        if self._server is None:
            raise RuntimeError(
                "WebsiteServer.bound_port is unavailable before start() "
                f"(requested port: {self.port})"
            )
        return int(self._server.server_address[1])

    def start(self) -> None:
        """Validate configuration, bind, and serve on a daemon thread."""
        if self._server is not None:
            raise RuntimeError(
                f"WebsiteServer is already serving on {self.url}; start() called twice"
            )
        bind_host = self._resolve_loopback_host(self.host)
        root = self.output_root
        if not root.is_dir():
            raise OutputRootNotFoundError(f"Output root not found: {root}")
        handler = functools.partial(
            _WebsiteRequestHandler,
            directory=str(root),
            live_reload=self.live_reload,
        )
        try:
            self._server = ThreadingHTTPServer((bind_host, self.port), handler)
        except OSError as exc:
            raise PortInUseError(
                f"Cannot bind website server on {bind_host}:{self.port}: {exc}"
            ) from exc
        self._bind_host = bind_host
        self._thread = threading.Thread(
            target=self._serve_forever,
            args=(self._server,),
            name="gnn-website-serve",
            daemon=True,
        )
        self._thread.start()
        logger.debug("Website server serving %s at %s", root, self.url)

    def shutdown(self) -> None:
        """Stop the server and join its thread; safe when never started."""
        if self._server is None:
            return
        server = self._server
        thread = self._thread
        self._server = None
        self._thread = None
        server.shutdown()
        server.server_close()
        if thread is not None:
            thread.join()

    def wait(self) -> None:
        """Block until the serving thread exits."""
        if self._thread is None:
            raise RuntimeError(
                "WebsiteServer.wait() requires a started server; call start() first "
                f"(output root: {self.output_root})"
            )
        self._thread.join()

    def _serve_forever(self, server: ThreadingHTTPServer) -> None:
        # Retain the bound server when immediate shutdown clears instance state.
        try:
            server.serve_forever()
        except OSError:
            logger.exception("Website server thread failed")

    @staticmethod
    def _resolve_loopback_host(host: str) -> str:
        """Return the loopback bind address for ``host``; refuse anything else."""
        if host == "localhost":
            return "127.0.0.1"
        try:
            addr = ipaddress.ip_address(host)
        except ValueError as exc:
            raise LoopbackViolationError(
                f"Refusing to bind host {host!r}: not a valid IP address ({exc})"
            ) from exc
        is_loopback = addr.version == 4 and addr in ipaddress.ip_network("127.0.0.0/8")
        is_loopback = is_loopback or (
            addr.version == 6 and addr == ipaddress.IPv6Address("::1")
        )
        if not is_loopback:
            raise LoopbackViolationError(
                f"Refusing to bind non-loopback host {host!r}: only 127.0.0.0/8 and "
                "::1 are permitted"
            )
        return str(addr)


def serve_website(
    output_root: str | Path,
    port: int = DEFAULT_WEBSITE_PORT,
    open_browser: bool = False,
    live_reload: bool = False,
    host: str = "127.0.0.1",
) -> None:
    """Serve the pipeline output tree over loopback HTTP until interrupted.

    Prints the landing-URL receipt, blocks until the serving thread exits
    (a ``KeyboardInterrupt`` in the caller thread shuts the server down
    cleanly and returns normally), then prints the stopped receipt.
    """
    server = WebsiteServer(output_root, port=port, host=host, live_reload=live_reload)
    server.start()
    landing = server.landing_url
    print(f"Serving pipeline output from {Path(output_root)} at {landing}")
    print("Press Ctrl-C to stop")
    try:
        if open_browser:
            webbrowser.open(landing)
        server.wait()
    except KeyboardInterrupt:
        logger.debug("Keyboard interrupt received, stopping website server")
    finally:
        server.shutdown()
    print(f"Website server stopped (was serving {landing})")
