"""Fail-closed reader contracts for the render-custody manifest.

``load_render_manifest`` is the shared reader behind the custody gates
(``scripts/z_record_manuscript_render_manifest.py``,
``scripts/z_verify_fresh_render.py``, and the strict token gate); every
failure must name the regen command instead of returning a partial dict.
``strip_volatile_tokens`` is the exact commit-varying token exclusion that
defines ``variables_sha256`` (mirrored by
``scripts/check_manuscript_tokens._strip_volatile_tokens``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.manuscript.render_custody import (
    DECLARED_ENVIRONMENT_KEY,
    STALE_CHAIN,
    TEMPLATE_REVISION_ENV,
    environment_report,
    load_render_manifest,
    manifest_path,
    strip_volatile_tokens,
    verify_fresh_render,
)

pytestmark = pytest.mark.unit

_MANIFEST_REL_PARTS = ("output", "data", "manuscript_render_manifest.json")


def _manifest_path(project_root: Path) -> Path:
    path = project_root.joinpath(*_MANIFEST_REL_PARTS)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_load_render_manifest_missing_names_regen_command(tmp_path: Path) -> None:
    """A missing manifest raises with the recording command, not a bare error."""
    with pytest.raises(RuntimeError, match="z_record_manuscript_render_manifest"):
        load_render_manifest(tmp_path)


def test_load_render_manifest_rejects_corrupt_json(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unreadable"):
        load_render_manifest(tmp_path)


def test_load_render_manifest_rejects_non_object(tmp_path: Path) -> None:
    path = _manifest_path(tmp_path)
    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        load_render_manifest(tmp_path)


def test_load_render_manifest_roundtrip_at_documented_path(tmp_path: Path) -> None:
    manifest = {
        "manifest_version": "gnn_render_manifest_v1",
        "counts_describe_commit": "abc1234",
    }
    path = _manifest_path(tmp_path)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert manifest_path(tmp_path) == path
    assert load_render_manifest(tmp_path) == manifest


def test_strip_volatile_tokens_drops_only_the_commit_token() -> None:
    variables = {
        "GNN_VERSION": "3.5.0",
        "GNN_GIT_COMMIT": "abc1234",
        "GNN_RELEASE_DATE": "2026-09-22",
    }
    assert strip_volatile_tokens(variables) == {
        "GNN_VERSION": "3.5.0",
        "GNN_RELEASE_DATE": "2026-09-22",
    }
    assert strip_volatile_tokens({}) == {}


def _write_manifest(project_root: Path, **extra: object) -> None:
    """Write a manifest carrying whatever declaration the case under test needs."""
    manifest = {
        "manifest_version": "gnn_render_manifest_v1",
        "counts_describe_commit": "abc1234",
        **extra,
    }
    _manifest_path(project_root).write_text(json.dumps(manifest), encoding="utf-8")


def test_template_gate_reports_a_moved_template_without_comparing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A moved template is one [STALE-CHAIN] finding, never an artifact verdict.

    The gate replaces the comparison: once the template has moved, every
    figure link in the hydrated prose is expected to differ, so an
    artifact-level message names the symptom and buries the cause — which is
    exactly what the 2026-11-03 red run did. ``tmp_path`` carries no render
    artifacts at all, so a returned artifact issue would prove the skip did
    not happen.
    """
    _write_manifest(
        tmp_path,
        **{
            DECLARED_ENVIRONMENT_KEY: {
                "template_revision": "75a91733f1124909d5bf639a1c74aeb7e3f712ff"
            }
        },
    )
    monkeypatch.setenv(
        TEMPLATE_REVISION_ENV, "7331ec06035384669a38036e02c52b2422d7f9d3"
    )
    issues = verify_fresh_render(tmp_path)
    assert len(issues) == 1, issues
    assert issues[0].startswith(STALE_CHAIN)
    assert "75a91733f1124909d5bf639a1c74aeb7e3f712ff" in issues[0]
    assert "7331ec06035384669a38036e02c52b2422d7f9d3" in issues[0]
    assert "re-record is required" in issues[0]


def test_template_gate_accepts_one_commit_at_another_abbreviation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nine characters and forty of one commit are not a move (git abbreviates)."""
    _write_manifest(
        tmp_path, **{DECLARED_ENVIRONMENT_KEY: {"template_revision": "7331ec060"}}
    )
    monkeypatch.setenv(
        TEMPLATE_REVISION_ENV, "7331ec06035384669a38036e02c52b2422d7f9d3"
    )
    assert not any(
        issue.startswith(STALE_CHAIN) for issue in verify_fresh_render(tmp_path)
    )


def test_template_gate_is_inert_without_a_declared_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No declaration means no gate; the verdict stays with the bytes."""
    _write_manifest(tmp_path)
    monkeypatch.setenv(
        TEMPLATE_REVISION_ENV, "7331ec06035384669a38036e02c52b2422d7f9d3"
    )
    assert not any(
        issue.startswith(STALE_CHAIN) for issue in verify_fresh_render(tmp_path)
    )


def test_environment_report_calls_a_legacy_manifest_unmeasured(tmp_path: Path) -> None:
    """A v1 chain declares nothing, and that is reported, never normalized."""
    _write_manifest(tmp_path)
    report = " ".join(environment_report(tmp_path, load_render_manifest(tmp_path)))
    assert "declares no environment (legacy v1)" in report
    assert "UNAVAILABLE" in report


def test_environment_report_reads_the_engine_banner_and_names_the_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine axis is read from the real log shape, and named when it moves."""
    log = tmp_path.joinpath("output", "pdf", "_combined_manuscript.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "This is XeTeX, Version 3.141592653-2.6-0.999998 (TeX Live 2026)\n"
        "(/usr/local/texlive/2026/texmf-dist/tex/latex/base/article.cls)\n",
        encoding="utf-8",
    )
    _write_manifest(
        tmp_path,
        **{
            DECLARED_ENVIRONMENT_KEY: {
                "template_revision": "",
                "tex_engine": "This is XeTeX, Version 3.141592653-2.6-0.999995",
                "tex_roots": ["/usr/share/texlive"],
            }
        },
    )
    monkeypatch.delenv(TEMPLATE_REVISION_ENV, raising=False)
    report = " ".join(environment_report(tmp_path, load_render_manifest(tmp_path)))
    assert (
        "tex_engine=This is XeTeX, Version 3.141592653-2.6-0.999998 "
        "(TeX Live 2026)" in report
    )
    assert "tex_roots=/usr/local/texlive/2026" in report
    assert "is unset: the template revision this render used is unmeasured" in report
    assert "NOT bounded yet" in report
