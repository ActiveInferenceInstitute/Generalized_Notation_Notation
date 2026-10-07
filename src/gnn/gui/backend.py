"""Shared GUI backend detection, artifact helpers, and launch verification."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast


@dataclass(frozen=True)
class GUIBackendStatus:
    """Runtime availability for an optional GUI backend."""

    name: Optional[str]
    module: Any
    reason: Optional[str] = None

    @property
    def available(self) -> bool:
        """Provide available behavior."""
        return self.name is not None


def detect_gradio_backend() -> GUIBackendStatus:
    """Return Gradio availability without failing default pipeline runs."""
    try:
        import gradio as gr

        if not hasattr(gr, "Blocks"):
            raise AttributeError("gradio import does not expose Blocks")
        return GUIBackendStatus(name="gradio", module=gr)
    except Exception as exc:
        return GUIBackendStatus(
            name=None,
            module=cast(Any, None),
            reason=f"{type(exc).__name__}: {exc}",
        )


def write_text_atomically(path: Path, content: str) -> None:
    """Write text via a temporary file in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp_f:
        tmp_f.write(content)
    os.replace(tmp_f.name, str(path))


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact via a temporary file in the destination directory."""
    write_text_atomically(path, json.dumps(payload, indent=2))


SERVER_POLL_ATTEMPTS = 30
SERVER_POLL_INTERVAL_S = 0.1


def wait_for_server_launch(
    thread: threading.Thread,
    port: int,
    *,
    attempts: int | None = None,
    interval: float | None = None,
) -> str | None:
    """Bounded launch verification for a just-launched server thread.

    Polls ``thread`` liveness every ``interval`` seconds, up to ``attempts``
    tries (default 30 x 0.1s = 3s), and probes ``http://127.0.0.1:<port>``
    to confirm the server actually answers. Returns ``None`` once the probe
    answers, or when the thread survived the whole bound while still starting
    up; otherwise returns a human-readable failure reason (the server thread
    exited before serving).
    """
    tries = SERVER_POLL_ATTEMPTS if attempts is None else attempts
    gap = SERVER_POLL_INTERVAL_S if interval is None else interval
    window = f"{tries * gap:.1f}s"
    for _ in range(tries):
        if not thread.is_alive():
            return f"server thread exited within {window} launch window"
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}",
                timeout=0.5,  # nosec B310
            ) as response:
                status = getattr(response, "status", None)
                if status is not None and status < 500:
                    return None
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                return None
        except OSError:
            pass
        time.sleep(gap)
    return None
