"""Exact delivery inventory and the template validation evidence that covers it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from gnn.manuscript.substitution import EXCLUDED_DOC_FILENAMES

PUBLICATION_PDF = Path("output/pdf/GeneralizedNotationNotation_combined.pdf")
_REPORT_ROOT = Path("output/reports")
_REPORT = _REPORT_ROOT / "validation_report.json"
_ARTIFACTS = _REPORT_ROOT / "artifact_manifest.json"
_PROVENANCE = _REPORT_ROOT / "rendered_provenance.json"
_EVIDENCE_FILES = {
    "validation_report": _REPORT,
    "artifact_manifest": _ARTIFACTS,
    "composition_manifest": _REPORT_ROOT / "manuscript_composition.json",
    "combined_manuscript": Path("output/web/_combined_manuscript.md"),
}

ArtifactDigests = dict[str, dict[str, str | int]]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"cannot read template validation evidence {path.name}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"template validation evidence {path.name} must be a JSON object"
        )
    return value


def _digests(project_root: Path, paths: set[Path]) -> ArtifactDigests:
    result: ArtifactDigests = {}
    for rel in sorted(paths):
        path = project_root / rel
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"publication artifact missing or not a regular file: {rel}"
            )
        content = path.read_bytes()
        if not content:
            raise RuntimeError(f"publication artifact is empty: {rel}")
        result[rel.as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    return result


def delivery_digests(project_root: Path) -> ArtifactDigests:
    """Hash the combined PDF and every enabled browser page, asset, and slide deck."""
    manuscript = project_root / "output/manuscript"
    config = project_root / "docs/manuscript/config.yaml"
    try:
        formats = yaml.safe_load(config.read_text(encoding="utf-8"))["render"][
            "formats"
        ]
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read manuscript render formats: {exc}") from exc
    if not isinstance(formats, dict) or formats.get("pdf") is not True:
        raise RuntimeError("manuscript render formats must enable the publication PDF")

    sections = {
        path.stem
        for path in manuscript.glob("*.md")
        if path.is_file() and path.name not in {*EXCLUDED_DOC_FILENAMES, "preamble.md"}
    }
    paths = {PUBLICATION_PDF}
    if formats.get("html"):
        web = Path("output/web")
        paths.add(web / "index.html")
        paths.update(web / f"manuscript__{stem}.html" for stem in sections)
        paths.update(
            path.relative_to(project_root)
            for path in (project_root / web / "assets").rglob("*")
            if path.is_file() or path.is_symlink()
        )
    if formats.get("slides"):
        paths.update(Path("output/slides") / f"{stem}_slides.pdf" for stem in sections)
    if (formats.get("html") or formats.get("slides")) and not sections:
        raise RuntimeError("publication has no hydrated sections to verify")
    return _digests(project_root, paths)


def validation_digests(
    project_root: Path, deliveries: ArtifactDigests
) -> ArtifactDigests:
    """Check Stage 04's verdict and bind its native receipt to the delivered bytes."""
    report = _read_object(project_root / _REPORT)
    checks = report.get("checks")
    summary = report.get("summary")
    if (
        not isinstance(summary, dict)
        or summary.get("all_passed") is not True
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError(
            "template Stage 04 validation must pass before recording publication custody"
        )

    provenance = _read_object(project_root / _PROVENANCE)
    evidence = provenance.get("evidence_files")
    if provenance.get(
        "schema_version"
    ) != "template-rendered-provenance-v2" or not isinstance(evidence, dict):
        raise RuntimeError(
            "template Stage 04 rendered provenance is missing its evidence bindings"
        )
    paths = {_PROVENANCE, *_EVIDENCE_FILES.values()}
    result = _digests(project_root, paths)
    for kind, rel in _EVIDENCE_FILES.items():
        if evidence.get(kind) != {"path": rel.as_posix(), **result[rel.as_posix()]}:
            raise RuntimeError(f"template Stage 04 provenance does not match {rel}")

    manifest = _read_object(project_root / _ARTIFACTS)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("template Stage 04 artifact manifest has no inventory")
    for delivery_path, digest in deliveries.items():
        matching = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("path") == delivery_path
        ]
        if (
            len(matching) != 1
            or matching[0].get("contract_match") is not True
            or any(matching[0].get(key) != value for key, value in digest.items())
        ):
            raise RuntimeError(
                f"template Stage 04 artifact inventory does not match {delivery_path}"
            )
    return result
