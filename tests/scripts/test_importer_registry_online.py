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
ORIGIN_HEAD = "a" * 40
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
        if args[:2] == ("api", "graphql"):
            assert "select(.isPrivate == false)" in args[-1]
            seen.append(f"graphql:{args[5]}")
            return f"{ORIGIN_HEAD}\n"
        url = args[1]
        seen.append(url)
        for suffix, body in contents.items():
            if url.endswith(f"{suffix}?ref={ORIGIN_HEAD}"):
                return body
        return None

    monkeypatch.setattr(check_importer_registry, "_gh", _gh)
    return seen


def test_online_reports_a_missing_pointer(monkeypatch) -> None:
    _stub(
        monkeypatch,
        {
            ".github/gnn-pin.json": PIN_BODY,
            ".github/workflows/ci.yml": WORKFLOW_BODY,
        },
    )
    findings = check_importer_registry.Findings()
    observations = check_importer_registry.check_online(
        _registry(_row("Alpha")), findings
    )
    assert observations == [
        {"repo": "Alpha", "revision": ORIGIN_HEAD, "state": "drift"}
    ]
    pointer = [item for item in findings.items if item["code"] == "origin-pointer"]
    assert len(pointer) == 1, findings.items
    assert "Alpha" in pointer[0]["message"]


def test_online_accepts_a_published_pointer(monkeypatch) -> None:
    seen = _stub(
        monkeypatch,
        {
            ".github/gnn-pin.json": PIN_BODY,
            ".github/workflows/ci.yml": WORKFLOW_BODY,
            ".github/GNN.md": "# GNN conformance\n",
        },
    )
    findings = check_importer_registry.Findings()
    observations = check_importer_registry.check_online(
        _registry(_row("Alpha")), findings
    )
    assert observations == [
        {"repo": "Alpha", "revision": ORIGIN_HEAD, "state": "matched"}
    ]
    base = "repos/ActiveInferenceInstitute/Alpha"
    assert seen == [
        "graphql:name=Alpha",
        f"{base}/contents/.github/gnn-pin.json?ref={ORIGIN_HEAD}",
        f"{base}/contents/.github/workflows/ci.yml?ref={ORIGIN_HEAD}",
        f"{base}/contents/.github/GNN.md?ref={ORIGIN_HEAD}",
    ]
    assert findings.items == []


def test_online_exempts_a_row_without_a_conformance_job(monkeypatch) -> None:
    seen = _stub(monkeypatch, {".github/gnn-pin.json": PIN_BODY})
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(
        _registry(_row("Alpha", conformance=False)), findings
    )
    assert not any("/contents/.github/GNN.md?" in url for url in seen)
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

    contents_calls: List[str] = []

    def _gh(*args: str) -> Optional[str]:
        if args[:2] == ("auth", "status"):
            return "logged in"
        if args[:2] == ("api", "graphql"):
            return f"{ORIGIN_HEAD}\n" if "name=Alpha" in args else ""
        contents_calls.append(args[1])
        raise check_importer_registry.GhError("HTTP 502 from api.github.com")

    monkeypatch.setattr(check_importer_registry, "_gh", _gh)
    findings = check_importer_registry.Findings()
    observations = check_importer_registry.check_online(
        _registry(_row("Alpha"), _row("Beta")), findings
    )
    assert observations == [
        {"repo": "Alpha", "revision": ORIGIN_HEAD, "state": "unavailable"},
        {"repo": "Beta", "revision": None, "state": "unavailable"},
    ]
    assert contents_calls == [
        f"repos/ActiveInferenceInstitute/Alpha/contents/.github/gnn-pin.json?ref={ORIGIN_HEAD}"
    ]
    assert [item["code"] for item in findings.items] == ["online-unavailable"] * 2
    assert findings.failed is False


def test_online_checks_the_job_and_pointer_for_a_dependency_pin(monkeypatch) -> None:
    """A dependency pin is read as text, and the job and pointer still run."""
    row = _row("Alpha")
    row["pin"] = {
        "kind": "dependency",
        "path": "pyproject.toml",
        "tag": "v5.0.3",
        "commit": COMMIT,
        "schema": "gnn-importer-pin/v1",
    }
    _stub(
        monkeypatch,
        {
            "pyproject.toml": (
                "generalized-notation-notation @ git+https://github.com/x/y@" + "0" * 40
            ),
            ".github/workflows/ci.yml": WORKFLOW_BODY,
        },
    )
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(_registry(row), findings)
    codes = [item["code"] for item in findings.items]
    assert "origin-pin-drift" in codes, findings.items
    assert "origin-pointer" in codes, findings.items
    assert "origin-job" not in codes, findings.items


def test_online_still_checks_the_job_and_pointer_without_the_pin_file(
    monkeypatch,
) -> None:
    """A missing pin is one finding; it must not hide the job and pointer."""
    _stub(monkeypatch, {".github/workflows/ci.yml": WORKFLOW_BODY})
    findings = check_importer_registry.Findings()
    check_importer_registry.check_online(_registry(_row("Alpha")), findings)
    codes = [item["code"] for item in findings.items]
    assert "origin-pin" in codes, findings.items
    assert "origin-pointer" in codes, findings.items
