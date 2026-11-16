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
        {"repo": "Alpha", "revision": ORIGIN_HEAD, "state": "drift", "reason": None}
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
        {"repo": "Alpha", "revision": ORIGIN_HEAD, "state": "matched", "reason": None}
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
        {
            "repo": "Alpha",
            "revision": ORIGIN_HEAD,
            "state": "unavailable",
            "reason": "transport",
        },
        {
            "repo": "Beta",
            "revision": None,
            "state": "unavailable",
            "reason": "transport",
        },
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


def test_auth_unavailable_covers_every_public_row(monkeypatch):
    monkeypatch.setattr(check_importer_registry, "_gh", lambda *args: None)
    findings = check_importer_registry.Findings()
    observations = check_importer_registry.check_online(
        _registry(_row("Alpha"), _row("Beta"), _row("Secret", visibility="private")),
        findings,
    )
    assert [o["repo"] for o in observations] == ["Alpha", "Beta"]
    assert all(
        o["state"] == "unavailable" and o["reason"] == "authentication"
        for o in observations
    )


def test_incomplete_observation_discards_partial_drift(monkeypatch):
    def gh(*args):
        if args[:2] == ("auth", "status"):
            return "ok"
        if args[:2] == ("api", "graphql"):
            return ORIGIN_HEAD
        if "gnn-pin.json" in args[1]:
            return '{"commit": "wrong", "tag": "wrong"}'
        raise check_importer_registry.GhError("unavailable", "permission")

    monkeypatch.setattr(check_importer_registry, "_gh", gh)
    findings = check_importer_registry.Findings()
    observations = check_importer_registry.check_online(
        _registry(_row("Alpha")), findings
    )
    assert not findings.failed
    assert observations[0]["reason"] == "permission"


def test_http_and_transport_failures_are_unavailable(monkeypatch):
    import subprocess

    for message, reason in [
        ("HTTP 401", "authentication"),
        ("HTTP 403", "permission"),
        ("HTTP 404", "http"),
        ("HTTP 502", "http"),
        ("connection reset", "transport"),
    ]:
        monkeypatch.setattr(
            check_importer_registry.subprocess,
            "run",
            lambda *a, message=message, **kw: subprocess.CompletedProcess(
                a, 1, "", message
            ),
        )
        try:
            check_importer_registry._gh("api", "test")
        except check_importer_registry.GhError as exc:
            assert exc.reason == reason
        else:
            raise AssertionError("failure accepted")


def test_category_schema_and_bidirectional_prose(monkeypatch, tmp_path):
    row = _row("Alpha")
    row["evidence_categories"] = ["source_pin", "conformance"]
    registry = _registry(row)
    registry["upstream"] = {"tag": "v5.0.3", "commit": COMMIT}
    findings = check_importer_registry.Findings()
    check_importer_registry.check_schema(registry, findings)
    assert not findings.failed
    prose = tmp_path / "README.md"
    prose.write_text("| [Alpha](alpha.md) | <!-- evidence_categories: source_pin -->\n")
    monkeypatch.setattr(check_importer_registry, "PROSE", prose)
    monkeypatch.setattr(
        check_importer_registry, "prose_repositories", lambda: {"Alpha"}
    )
    check_importer_registry.check_docs_parity(registry, findings)
    assert any(i["code"] == "docs-evidence-categories" for i in findings.items)
    for categories in [[], ["bogus"], ["source_pin", "source_pin"], [dict()]]:
        row["evidence_categories"] = categories
        findings = check_importer_registry.Findings()
        check_importer_registry.check_schema(registry, findings)
        assert any(i["code"] == "row-evidence-categories" for i in findings.items)


def test_spawn_and_timeout_failures_are_structured(monkeypatch):
    import subprocess

    for failure in [FileNotFoundError(), subprocess.TimeoutExpired(["gh"], 1)]:

        def fail(*args, failure=failure, **kwargs):
            raise failure

        monkeypatch.setattr(check_importer_registry.subprocess, "run", fail)
        try:
            check_importer_registry._gh("auth", "status")
        except check_importer_registry.GhError as exc:
            assert exc.reason == "transport"
        else:
            raise AssertionError("failed process was accepted")


def test_cli_distinguishes_unavailable_from_failed(monkeypatch, capsys):
    import json

    monkeypatch.setattr(
        check_importer_registry, "load_registry", lambda: _registry(_row("Alpha"))
    )
    for name in [
        "check_schema",
        "check_docs_parity",
        "check_pin_parity",
        "check_workflow_groups",
    ]:
        monkeypatch.setattr(check_importer_registry, name, lambda *args: None)
    for state, code, status in [
        ("matched", 0, "passed"),
        ("unavailable", 2, "unavailable"),
        ("drift", 1, "failed"),
    ]:

        def online(registry, findings, state=state):
            if state == "drift":
                findings.error("origin-pin-drift", "declared pin differs")
            return [
                {
                    "repo": "Alpha",
                    "revision": ORIGIN_HEAD,
                    "state": state,
                    "reason": "http" if state == "unavailable" else None,
                }
            ]

        monkeypatch.setattr(check_importer_registry, "check_online", online)
        assert check_importer_registry.main(["--online", "--json"]) == code
        report = json.loads(capsys.readouterr().out)
        assert report["status"] == status
        assert report["ok"] == (state == "matched")
