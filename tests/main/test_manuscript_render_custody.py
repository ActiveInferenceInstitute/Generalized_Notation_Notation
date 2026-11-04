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

import gnn.manuscript.render_custody as render_custody
from gnn.manuscript.render_custody import (
    DECLARED_ENVIRONMENT_KEY,
    PERMITTED_PROFILES_KEY,
    PERMITTED_RENDER_PROFILES,
    PROFILE_KEYS,
    RECORDING_PROFILE_KEY,
    STALE_CHAIN,
    TEMPLATE_REVISION_ENV,
    UNMEASURED,
    comparison_scope,
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


def test_environment_report_calls_a_v1_manifest_unmeasured(tmp_path: Path) -> None:
    """A v1 chain declares nothing, and that is reported, never normalized."""
    _write_manifest(tmp_path)
    report = " ".join(environment_report(tmp_path, load_render_manifest(tmp_path)))
    assert "declares no environment (v1 manifest)" in report
    assert "UNAVAILABLE" in report


def test_environment_report_reads_the_engine_banner_and_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine axis is read from the real log shape, and the set is named."""
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
                RECORDING_PROFILE_KEY: "recording",
                PERMITTED_PROFILES_KEY: {
                    "recording": {
                        "pandoc": "3.13",
                        "pandoc_crossref": "0.3.26",
                        "tex_engine": (
                            "This is XeTeX, Version 3.141592653-2.6-0.999998 "
                            "(TeX Live 2026)"
                        ),
                        "tex_roots": ["/usr/local/texlive/2026"],
                    }
                },
            }
        },
    )
    monkeypatch.setattr(
        render_custody,
        "_probe_toolchain",
        lambda: {"pandoc": "3.13", "pandoc_crossref": "0.3.26"},
    )
    monkeypatch.delenv(TEMPLATE_REVISION_ENV, raising=False)
    report = " ".join(environment_report(tmp_path, load_render_manifest(tmp_path)))
    assert (
        "tex_engine=This is XeTeX, Version 3.141592653-2.6-0.999998 "
        "(TeX Live 2026)" in report
    )
    assert "tex_roots=/usr/local/texlive/2026" in report
    assert "is unset: the template revision this render used is unmeasured" in report
    assert "recording profile 'recording'" in report
    assert "permitted profiles (1): recording" in report
    assert "comparing 3 rendered artifacts" in report


_PROBE = {"pandoc": "3.13", "pandoc_crossref": "0.3.26"}
_LOCAL_ENGINE = "This is XeTeX, Version 3.141592653-2.6-0.999998 (TeX Live 2026)"
_CI_PROFILE_NAME, _CI_PROFILE = next(iter(PERMITTED_RENDER_PROFILES.items()))
_LOCAL_PROFILE: dict[str, object] = {
    "pandoc": "3.13",
    "pandoc_crossref": "0.3.26",
    "tex_engine": _LOCAL_ENGINE,
    "tex_roots": ["/usr/local/texlive/2026"],
}
_LOCAL_LOG = (
    f"{_LOCAL_ENGINE} (preloaded format=xelatex 2026.9.18)  4 NOV 2026 12:32\n"
    "(/usr/local/texlive/2026/texmf-dist/tex/latex/base/article.cls)\n"
)
_CI_LOG = (
    "This is XeTeX, Version 3.141592653-2.6-0.999995 (TeX Live 2023/Debian) "
    "(preloaded format=xelatex 2026.11.4)  4 NOV 2026 15:37\n"
    "(/usr/share/texlive/texmf-dist/tex/latex/base/article.cls)\n"
)


def _profile_tree(tmp_path: Path, log_text: str) -> dict[str, Path]:
    """A render tree whose comparison the profile decision can be asked about."""
    pdf = tmp_path / "output" / "pdf"
    pdf.mkdir(parents=True, exist_ok=True)
    paths = {
        "log": pdf / "_combined_manuscript.log",
        "md": pdf / "_combined_manuscript.md",
        "tex": pdf / "_combined_manuscript.tex",
    }
    paths["log"].write_text(log_text, encoding="utf-8")
    paths["md"].write_text("prose\n", encoding="utf-8")
    paths["tex"].write_text("\\begin{document}\n\n\\end{document}\n", encoding="utf-8")
    return paths


def _digests(paths: dict[str, Path]) -> dict[str, str]:
    """The digests a record step would write for this tree, normalized.

    Built through the module's own digest function so the fixture agrees with
    the comparison: the ``.log``'s banner timestamp is masked by the
    comparison and must be masked here too, or the fixture would be testing
    the recorder's normalization rather than the profile decision.
    """
    stamps = ("abc1234",)
    return {
        rel: render_custody._artifact_digest_from_bytes(rel, path.read_bytes(), stamps)
        for rel, path in ((f"output/pdf/{path.name}", path) for path in paths.values())
    }


def _profile_manifest(
    tmp_path: Path,
    *,
    digests: dict[str, str],
    recording: dict[str, object] | None = None,
    template_revision: str = "abc1234",
) -> None:
    """A manifest that declares one recording profile and permits the CI one."""
    recorded = dict(recording if recording is not None else _LOCAL_PROFILE)
    _write_manifest(
        tmp_path,
        **{
            "render_artifacts_sha256": digests,
            "render_artifacts_normalized_sha256": digests,
            "render_inputs_sha256": {},
            DECLARED_ENVIRONMENT_KEY: {
                "template_revision": template_revision,
                RECORDING_PROFILE_KEY: "recording",
                PERMITTED_PROFILES_KEY: {
                    "recording": recorded,
                    _CI_PROFILE_NAME: dict(_CI_PROFILE),
                },
            },
        },
    )


def test_a_permitted_non_recording_profile_narrows_the_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine-dependent bytes are excluded, named and counted; the rest is not.

    The `.log` is TeX Live's own output and TeX Live is the one axis a
    permitted profile may differ in, so its bytes are not compared on a host
    that is not the recording one — and the `.md`/`.tex` under it still are,
    which the second half proves by drifting the `.tex` and getting a failure.
    """
    monkeypatch.setattr(render_custody, "_probe_toolchain", lambda: dict(_PROBE))
    monkeypatch.setenv(TEMPLATE_REVISION_ENV, "abc1234")
    paths = _profile_tree(tmp_path, _CI_LOG)
    digests = _digests(paths)
    digests["output/pdf/_combined_manuscript.log"] = "0" * 64  # never compared here
    _profile_manifest(tmp_path, digests=digests)
    assert verify_fresh_render(tmp_path) == []
    scope = comparison_scope(tmp_path, load_render_manifest(tmp_path))
    assert "2 of 3 rendered artifacts" in scope
    assert "_combined_manuscript.log not compared on this host" in scope

    paths["tex"].write_text(
        "\\begin{document}\n% drifted\n\\end{document}\n", encoding="utf-8"
    )
    issues = verify_fresh_render(tmp_path)
    assert len(issues) == 1, issues
    assert issues[0].startswith("[FAIL] output/pdf/_combined_manuscript.tex"), issues


def test_a_profile_outside_the_permitted_set_fails_without_comparing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pandoc and pandoc-crossref are pinned; a third version is a drifted pin.

    The recorded digests are deliberately wrong, so a `.tex` finding would
    prove the comparison ran: bytes from an undeclared toolchain are not
    evidence about this chain and must not be reported as if they were.
    """
    monkeypatch.setattr(
        render_custody,
        "_probe_toolchain",
        lambda: {"pandoc": "3.14", "pandoc_crossref": "0.3.26"},
    )
    monkeypatch.setenv(TEMPLATE_REVISION_ENV, "abc1234")
    paths = _profile_tree(tmp_path, _CI_LOG)
    _profile_manifest(tmp_path, digests=_digests(paths))
    issues = verify_fresh_render(tmp_path)
    assert len(issues) == 1, issues
    assert issues[0].startswith("[FAIL] render profile"), issues
    assert "pandoc='3.14'" in issues[0]
    assert "artifact comparison is skipped" in issues[0]
    assert _CI_PROFILE_NAME in issues[0]


def test_an_unmeasured_template_revision_is_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch is off: unmeasured is its own status, never a pass (exit 4)."""
    monkeypatch.setattr(render_custody, "_probe_toolchain", lambda: dict(_PROBE))
    paths = _profile_tree(tmp_path, _LOCAL_LOG)
    _profile_manifest(tmp_path, digests=_digests(paths))
    monkeypatch.delenv(TEMPLATE_REVISION_ENV, raising=False)
    issues = verify_fresh_render(tmp_path)
    assert len(issues) == 1, issues
    assert issues[0].startswith(UNMEASURED), issues
    assert "abc1234" in issues[0]


def test_artifact_drift_with_matching_inputs_fails_not_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same prose, different bytes, permitted profile: drift has no warning level."""
    monkeypatch.setattr(render_custody, "_probe_toolchain", lambda: dict(_PROBE))
    monkeypatch.setenv(TEMPLATE_REVISION_ENV, "abc1234")
    paths = _profile_tree(tmp_path, _LOCAL_LOG)
    _profile_manifest(tmp_path, digests=_digests(paths))
    paths["tex"].write_text(
        "\\begin{document}\n% drifted\n\\end{document}\n", encoding="utf-8"
    )
    issues = verify_fresh_render(tmp_path)
    assert len(issues) == 1, issues
    assert issues[0].startswith("[FAIL] output/pdf/_combined_manuscript.tex"), issues
    assert "the render inputs match this chain" in issues[0]
    assert not any("[WARN]" in issue for issue in issues)


def test_no_artifact_comparison_warning_level_remains() -> None:
    """A normalized-vs-normalized mismatch is drift; the level must not exist."""
    source = Path(render_custody.__file__).read_text(encoding="utf-8")
    assert "[WARN]" not in source


def test_every_profile_key_is_one_of_the_declared_ones() -> None:
    """The compared axes are exactly the declared ones — no axis creeps in."""
    assert set(PROFILE_KEYS) == {"pandoc", "pandoc_crossref", "tex_engine", "tex_roots"}
    for name, values in PERMITTED_RENDER_PROFILES.items():
        assert set(values) == set(PROFILE_KEYS), name
