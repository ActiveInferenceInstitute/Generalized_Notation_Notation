#!/usr/bin/env python3
"""The online registry re-read must also confirm the GNN pointer exists.

``docs/downstream/README.md`` requires every repository that publishes a
conformance job to carry ``.github/GNN.md`` recording what that job parses.
The offline gate cannot see origin, so ``check_online`` re-reads it. These
tests pin that behaviour with a stubbed ``gh``, so no network is touched:

- a row whose repository publishes the pointer reports nothing;
- a row whose repository lacks it reports one ``origin-pointer`` error;
- a row without a conformance job never asks for the pointer;
- a private row is skipped entirely;
- an unauthenticated ``gh`` degrades to the existing ``online-unavailable``
  warning instead of a false finding.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "check_importer_registry.py"
_spec = importlib.util.spec_from_file_location("check_importer_registry", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_importer_registry = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("check_importer_registry", check_importer_registry)
_spec.loader.exec_module(check_importer_registry)

COMMIT = "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1"
PIN = {
    "kind": "file",
    "path": ".github/gnn-pin.json",
    "tag": "v5.0.3",
    "commit": COMMIT,
    "schema": "gnn-importer-pin/v1",
}
PIN_BODY = '{"commit": "%s", "tag": "v5.0.3"}' % COMMIT
WORKFLOW_BODY = "jobs:\n  gnn-conformance:\n    name: gnn conformance (pinned)\n"


def _row(
    repo: str, *, visibility: str = "public", conformance: bool = True
) -> Dict[str, Any]:
    return {
        "repo": repo,
        "visibility": visibility,
        "kind": "spec",
        "pin": dict(PIN),
        "specs": {"count": 1},
        "conformance": (
            {
                "workflow": ".github/workflows/ci.yml",
                "job": "gnn-conformance",
                "name": "gnn conformance (pinned)",
            }
            if conformance
            else None
        ),
    }


def _registry(*rows: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema": "gnn-importer-registry/v1", "importers": list(rows)}


def _stub(monkeypatch, contents: Dict[str, str]) -> List[str]:
    """Install a ``gh`` stub; return the list it records requested URLs in."""
    seen: List[str] = []

    def _gh(*args: str) -> Optional[str]:
        if args[:2] == ("auth", "status"):
            return "logged in"
        url = args[1]
        seen.append(url)
        if len(args) > 2 and args[2] == "--jq":
            return "main\n"
        for suffix, body in contents.items():
            if url.endswith(suffix):
                return body
        return None

    monkeypatch.setattr(check_importer_registry, "_gh", _gh)
    return seen


def test_online_reports_a_missing_pointer(monkeypatch) -> None:
    _stub(
        monkeypatch,
        {
            ".github/gnn-pin.json?ref=main": PIN_BODY,
            ".github/workflows/ci.yml?ref=main": WORKFLOW_BODY,
        },
    )
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(_registry(_row("Alpha")), findings)
    pointer = [item for item in findings.items if item["code"] == "origin-pointer"]
    assert len(pointer) == 1, findings.items
    assert "Alpha" in pointer[0]["message"]


def test_online_accepts_a_published_pointer(monkeypatch) -> None:
    _stub(
        monkeypatch,
        {
            ".github/gnn-pin.json?ref=main": PIN_BODY,
            ".github/workflows/ci.yml?ref=main": WORKFLOW_BODY,
            ".github/GNN.md?ref=main": "# GNN conformance\n",
        },
    )
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(_registry(_row("Alpha")), findings)
    assert findings.items == []


def test_online_exempts_a_row_without_a_conformance_job(monkeypatch) -> None:
    seen = _stub(monkeypatch, {".github/gnn-pin.json?ref=main": PIN_BODY})
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(
        _registry(_row("Alpha", conformance=False)), findings
    )
    assert not any(url.endswith(".github/GNN.md?ref=main") for url in seen)
    assert findings.items == []


def test_online_skips_a_private_row(monkeypatch) -> None:
    seen = _stub(monkeypatch, {})
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(
        _registry(_row("Alpha", visibility="private")), findings
    )
    assert seen == []
    assert findings.items == []


def test_online_is_unavailable_without_gh(monkeypatch) -> None:
    monkeypatch.setattr(check_importer_registry, "_gh", lambda *args: None)
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(_registry(_row("Alpha")), findings)
    assert [item["code"] for item in findings.items] == ["online-unavailable"]


def test_online_degrades_a_transient_failure_to_a_warning(monkeypatch) -> None:
    """A 5xx or a timeout is not evidence that a file is missing."""

    def _gh(*args: str) -> Optional[str]:
        if args[:2] == ("auth", "status"):
            return "logged in"
        if len(args) > 2 and args[2] == "--jq":
            return "main\n"
        raise check_importer_registry.GhError("HTTP 502 from api.github.com")

    monkeypatch.setattr(check_importer_registry, "_gh", _gh)
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(_registry(_row("Alpha")), findings)
    assert [item["code"] for item in findings.items] == ["online-unavailable"]
    assert findings.failed is False
