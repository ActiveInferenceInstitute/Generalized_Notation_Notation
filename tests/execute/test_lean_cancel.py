#!/usr/bin/env python3
"""
Deterministic CancelToken pin for the lean dispatch route (M-11, W7-M11).

The lean execution route must thread the cooperative ``CancelToken`` through
every public hop: ``run_lean_scripts`` (batch entry) → ``verify_document`` →
``_verify_document_impl`` → ``run_subprocess_envelope``. The envelope owns
the cooperative checks (pre-spawn and every poll slice); the fep-lean bridge
process itself has no in-process cooperative-cancel protocol yet (held
fep-side substance), so GNN-side cancellation is enforced at the envelope
boundary.

Every test is deterministic and zero-skip: the fep_lean root resolution and
the subprocess envelope are stubbed, or the token is pre-fired so the real
envelope cancels before spawning. No real ``uv``/fep invocation, no sleeps,
no ``needs_lean`` markers.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute.lean import lean_runner
from gnn.execute.subprocess_envelope import NEVER_STARTED, CancelToken


def _pin_fake_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Resolve FEP_Formal to a stub checkout so the envelope call is reached."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text('[tool.uv]\nrequired-version = "==0.13.0"\n')
    (root / ".python-version").write_text("3.14\n")
    (root / ".venv").mkdir()
    (root / ".venv/pyvenv.cfg").write_text("version_info = 3.14.0\n")
    monkeypatch.setattr(lean_runner, "resolve_fep_formal_root", lambda: root)


def _cancelled_envelope() -> Dict[str, Any]:
    """The envelope a pre-spawn cancel produces (real keys, no run)."""
    return {
        "success": False,
        "return_code": NEVER_STARTED,
        "stdout": "",
        "stderr": "",
        "error": "Execution cancelled: unit-test",
        "error_type": "Cancelled",
        "cancelled": True,
        "duration_seconds": 0.0,
        "sandbox_mode": "off",
        "sandboxed": False,
    }


def _write_document(directory: Path, stem: str) -> Path:
    """Emit a discoverable GNN document (``_is_gnn_document`` passes)."""
    document = directory / f"{stem}.md"
    document.write_text("## GNNSection\nstate: s\n", encoding="utf-8")
    return document


def test_prespawn_cancel_reports_cancellation_and_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-fired token reaches the envelope; the record reports cancellation."""
    calls: List[Dict[str, Any]] = []

    def fake_envelope(command: List[str], **kwargs: Any) -> Dict[str, Any]:
        calls.append({"command": command, **kwargs})
        return _cancelled_envelope()

    _pin_fake_root(monkeypatch, tmp_path / "fep_lean")
    monkeypatch.setattr(lean_runner, "run_subprocess_envelope", fake_envelope)

    document = _write_document(tmp_path, "agent")
    token = CancelToken()
    token.cancel("unit-test")

    record = lean_runner.verify_document(
        document, tmp_path / "receipt.json", cancel_token=token
    )

    assert len(calls) == 1, f"expected one envelope call; got {calls}"
    assert calls[0]["cancel_token"] is token, "token object must reach the envelope"
    assert calls[0]["env"]["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "fep_lean/.venv")
    assert "uv==0.13.0" in calls[0]["command"]
    assert record["success"] is False
    assert record["cancelled"] is True
    assert record["error_type"] == "Cancelled"
    assert record["return_code"] == NEVER_STARTED
    # The stub replaces the spawn boundary: nothing ran and no bridge
    # process could have produced a receipt.
    assert not (tmp_path / "receipt.json").exists()


def test_token_threads_through_public_chain_to_impl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``verify_document`` forwards ``cancel_token`` into ``_verify_document_impl``."""
    seen: Dict[str, Any] = {}

    def fake_impl(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        seen["positional"] = args
        seen["kwargs"] = kwargs
        return {"success": False, "error": "stubbed impl"}

    _pin_fake_root(monkeypatch, tmp_path / "fep_lean")
    monkeypatch.setattr(lean_runner, "_verify_document_impl", fake_impl)

    token = CancelToken()
    document = _write_document(tmp_path, "agent")
    record = lean_runner.verify_document(
        document, tmp_path / "receipt.json", cancel_token=token
    )

    assert record == {"success": False, "error": "stubbed impl"}
    assert seen["kwargs"]["cancel_token"] is token
    assert seen["positional"][0] == Path(document).resolve()


def test_run_lean_scripts_threads_token_to_every_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch entry hands the same token object to every per-document hop."""
    calls: List[Dict[str, Any]] = []

    def fake_envelope(command: List[str], **kwargs: Any) -> Dict[str, Any]:
        calls.append({"command": command, **kwargs})
        return _cancelled_envelope()

    _pin_fake_root(monkeypatch, tmp_path / "fep_lean")
    monkeypatch.setattr(lean_runner, "run_subprocess_envelope", fake_envelope)

    target = tmp_path / "target"
    target.mkdir()
    _write_document(target, "doc_a")
    _write_document(target, "doc_b")

    token = CancelToken()
    token.cancel("unit-test")
    ok = lean_runner.run_lean_scripts(
        target, tmp_path / "lean_out", recursive_search=True, cancel_token=token
    )

    assert ok is False
    assert len(calls) == 2, f"one envelope call per document; got {len(calls)}"
    assert all(call["cancel_token"] is token for call in calls)


def test_prespawn_cancel_with_real_envelope_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real envelope a pre-fired token cancels before any spawn."""
    # Sandbox resolution precedes the pre-spawn cancel check inside the
    # envelope; clear GNN_SANDBOX so the SandboxUnavailable refusal cannot
    # shadow the cancellation on a require-mode host.
    monkeypatch.delenv("GNN_SANDBOX", raising=False)
    _pin_fake_root(monkeypatch, tmp_path / "fep_lean")

    document = _write_document(tmp_path, "agent")
    token = CancelToken()
    token.cancel("pre-spawn")

    record = lean_runner.verify_document(
        document, tmp_path / "receipt.json", cancel_token=token
    )

    assert record["success"] is False
    assert record["cancelled"] is True
    assert record["error_type"] == "Cancelled"
    assert record["return_code"] == NEVER_STARTED
    assert "cancel" in record["error"].lower()
    assert not (tmp_path / "receipt.json").exists()


def test_default_none_keeps_existing_paths_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No token: both signatures default to None and records stay cancel-free."""
    for name in ("verify_document", "run_lean_scripts"):
        parameter = inspect.signature(getattr(lean_runner, name)).parameters[
            "cancel_token"
        ]
        assert parameter.default is None, f"{name}.cancel_token must default to None"

    calls: List[Dict[str, Any]] = []

    def fake_envelope(command: List[str], **kwargs: Any) -> Dict[str, Any]:
        calls.append({"command": command, **kwargs})
        return {
            "success": True,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0.0,
            "cancelled": False,
            "sandbox_mode": "off",
            "sandboxed": False,
        }

    _pin_fake_root(monkeypatch, tmp_path / "fep_lean")
    monkeypatch.setattr(lean_runner, "run_subprocess_envelope", fake_envelope)

    document = _write_document(tmp_path, "agent")
    record = lean_runner.verify_document(document, tmp_path / "receipt.json")

    assert calls[0]["cancel_token"] is None
    assert record["success"] is True
    assert "cancelled" not in record, "default path must not grow cancel fields"
    assert "error_type" not in record
